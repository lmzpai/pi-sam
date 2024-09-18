import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from segment_anything.dataset import create_dataloaders
from segment_anything.utils import dist, misc
from segment_anything import sam_model_registry, build_mask_decoder
from tensorboardX import SummaryWriter
from PIL import Image
from tqdm import tqdm


class Runner:
    def __init__(self, logger, args) -> None:
        self.logger = logger
        self.out_dir = args.output
        self.logger.info("--- build sam ---")
        sam_encoder = sam_model_registry[args.model_type](checkpoint=args.sam_ckpt)
        self.sam_encoder = sam_encoder.to(dist.dev())
        self.logger.info(f"--- build sam: {args.model_type}, load from {args.sam_ckpt} ---")

        self.logger.info("--- build new mask_deocder ---")
        net = build_mask_decoder(args.decoder_type, args.model_type, args.decoder_ckpt)
        self.net = net.to(dist.dev())
        self.logger.info(f"--- build mask decoder: {args.decoder_type}, load from {args.decoder_ckpt} ---")

        self.training = not args.eval

        if self.training:
            self._train_init(args)
        else:
            self._test_init(args)
             
    def _train_init(self, args):
        ### --- Step 1: Train Dataloader ---
        self.logger.info("--- create training dataloader ---")
        train_dataloaders, _ = create_dataloaders(
            args.train_used_dataset,
            args.data_root,
            self.logger,
            batch_size = args.batch_size_train,
            training = True)
        self.steps_per_epoch = len(train_dataloaders)
        self.logger.info(f"{self.steps_per_epoch} train dataloaders created")
        self.train_dataloaders = train_dataloaders

        if dist.is_dist_avail_and_initialized():
            self.net = torch.nn.parallel.DistributedDataParallel(self.net, find_unused_parameters=args.find_unused_params)
            self.net_without_ddp = self.net.module
        else:
            self.net_without_ddp = self.net
        
        self.logger.info("--- define optimizer ---")
        train_params = self.get_train_params()
        self.optimizer = optim.Adam(train_params, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.max_epoch_num)

        if args.restore_model:
            ckpt = torch.load(args.restore_model,map_location="cpu")['model_state_dict']
            self.net_without_ddp.load_state_dict(ckpt, strict=False)      
            self.lr_scheduler.last_epoch = args.start_epoch - 1

        self.epoch_start = args.start_epoch
        self.epoch_num = args.max_epoch_num
        self.model_save_fre = args.model_save_fre
        self.log_print_fre = args.log_print_fre
        self.no_validate = args.no_validate
        self._init_tensorboard(output_path = os.path.join(self.out_dir, 'tensorbd'))

        if not args.no_validate:
            self.logger.info("--- create valid dataloader ---")
            valid_dataloaders, valid_datasets = create_dataloaders(
                args.valid_used_dataset,
                args.data_root,
                self.logger,
                batch_size=args.batch_size_valid,
                training=False)
            self.logger.info(f"{len(valid_dataloaders)} valid dataloaders created")
            self.valid_dataloaders = valid_dataloaders
            self.valid_datasets = valid_datasets
    
    def get_train_params(self):
        train_params = []
        for n, p in self.net_without_ddp.named_parameters():
            if p.requires_grad:
                train_params.append(p)
        return train_params

    def _test_init(self, args):
        self.logger.info("--- create test dataloader ---")
        test_dataloaders, _ = create_dataloaders(
            args.test_used_dataset,
            args.data_root,
            self.logger,
            batch_size=args.batch_size_valid,
            training=False)
        self.logger.info(f"{len(test_dataloaders)} test dataloaders created")
        self.net_without_ddp = self.net
        self.test_dataloaders = test_dataloaders
        assert args.restore_model is not None
        ckpt = torch.load(args.restore_model,map_location="cpu")
        if "model_state_dict" in ckpt:
            self.net_without_ddp.load_state_dict(ckpt["model_state_dict"])
        else:
            self.net_without_ddp.load_state_dict(ckpt) 
    
    def _init_tensorboard(self, output_path=None):
        if output_path is None:
            output_path = self.out_dir
        if dist.get_rank() == 0:
            self.tensorboard_writer = SummaryWriter(logdir=output_path)
            
    def train(self):            
        for epoch in range(self.epoch_start, self.epoch_num): 
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(f"------------------------- new epoch -------------------------")
            self.logger.info(f"------------ epoch: {epoch}, learning rate: {lr} ------------")
            if dist.is_dist_avail_and_initialized():
                self.train_dataloaders.batch_sampler.sampler.set_epoch(epoch)
            for step, data in enumerate(self.train_dataloaders):
                # for  in enumerate(dataloader):
                inputs, labels = data['image'], data['label']
                inputs = inputs.to(dist.dev())
                labels = labels.to(dist.dev())
                self.train_one_step(inputs, labels, step, epoch)
            
            self.logger.info(f"------------------- Finish epoch: {epoch} -------------------")
            self.lr_scheduler.step()
            if epoch % self.model_save_fre == 0:
                self.save(epoch)
                if not self.no_validate:
                    self.net.eval()
                    self.val(epoch)
                    self.net.train()


    def train_one_step(self, inputs, labels, step, epoch):
        labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
        batched_input = self.get_batched_input(inputs, labels, labels_256, input_keys = ['box'])
        with torch.no_grad():
            encoder_outputs, interm_embeddings = self.sam_encoder(batched_input, multimask_output=False)
        decoder_outputs = self.net(encoder_outputs, interm_embeddings, labels / 255.0, labels_256, interact=True, get_loss=True)
        loss, info_dict = self.loss_post_process(decoder_outputs['losses'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if (step + 1) % self.log_print_fre == 0:
            if 'pred_masks' in decoder_outputs:
                info_dict['iou'] = compute_iou(decoder_outputs['pred_masks'], labels_256*255)
                info_dict['bnd_iou'] = compute_boundary_iou(decoder_outputs['pred_masks'], labels_256*255)
            if 'pred_masks_large' in decoder_outputs:
                info_dict['iou_l'] = compute_iou(decoder_outputs['pred_masks_large'], labels)
                info_dict['bnd_iou_l'] = compute_boundary_iou(decoder_outputs['pred_masks_large'], labels)
            if 'pred_masks_pi' in decoder_outputs:
                info_dict['iou_pi'] = compute_iou(decoder_outputs['pred_masks_pi'], labels)
                info_dict['bnd_iou_pi'] = compute_boundary_iou(decoder_outputs['pred_masks_pi'], labels)
            self._log_info(info_dict, step+1, epoch)
            self._write_tensorboard(info_dict, epoch*self.steps_per_epoch+step+1)
    

    def get_batched_input(self, inputs, labels, labels_256=None, 
                          input_keys = ['box','point','noise_mask']):
        labels_box = misc.masks_to_boxes(labels[:,0,:,:])
        if 'point' in input_keys:
            try:
                labels_points = misc.masks_sample_points(labels[:,0,:,:])
            except:
                # less than 10 points
                input_keys.remove('point')
        if 'noise_mask' in input_keys:
            labels_noisemask = misc.masks_noise(labels_256)
        batched_input = []
        for b_i, image in enumerate(inputs):
            dict_input = dict()
            dict_input['image'] = image 
            input_type = random.choice(input_keys)
            if input_type == 'box':
                dict_input['boxes'] = labels_box[b_i:b_i+1]
            elif input_type == 'point':
                point_coords = labels_points[b_i:b_i+1]
                dict_input['point_coords'] = point_coords
                dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
            elif input_type == 'noise_mask':
                dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
            else:
                raise NotImplementedError
            dict_input['original_size'] = (1024, 1024)
            batched_input.append(dict_input)
        return batched_input
    

    def loss_post_process(self, outputs):
        log_vars = dict()
        losses = []
        for k, v in outputs.items():
            if 'loss' in k:
                losses.append(v)
            log_vars[k] = v.item()
        
        assert len(losses) > 0
        losses = sum(losses)
        log_vars['losses'] = losses.item()
        return losses, log_vars
    

    def _write_tensorboard(self, info_dict, global_step):
        info_dict = dist.reduce_dict(info_dict)
        if dist.get_rank() == 0:
            for k, v in info_dict.items():
                self.tensorboard_writer.add_scalar(k, v, global_step=global_step)


    def _log_info(self, info_dict, step, epoch):
        self.logger.info('----------------------------------')
        self.logger.info(f'epoch: {epoch}, step: {step} / {self.steps_per_epoch}')
        for k, v in info_dict.items():
            self.logger.info(f'{k}: {v}')
    
    def save(self, epoch):
        if dist.get_rank() == 0:
            ckpt_dict = dict(
                model_state_dict = self.net_without_ddp.state_dict())
            ckpt_file = os.path.join(self.out_dir, f'{epoch}.pth')
            torch.save(ckpt_dict, ckpt_file)
            self.logger.info(f"saving epoch: {epoch} done!")
        if dist.is_dist_avail_and_initialized():
            torch.distributed.barrier()
    
    @torch.no_grad()
    def val(self, epoch):        
        # ious_pi, bnd_ious_pi = [], []
        for dataloader, dataset_name in zip(self.valid_dataloaders, self.valid_datasets):
            ious, bnd_ious = [], []
            ious_l, bnd_ious_l = [], []
            for data in dataloader:
                inputs, labels = data['image'], data['label']
                inputs = inputs.to(dist.dev())
                labels = labels.to(dist.dev())
                labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
                batched_input = self.get_batched_input(inputs, labels, input_keys = ['box'])
                encoder_outputs, interm_embeddings = self.sam_encoder(batched_input, multimask_output=False)
                decoder_outputs = self.net(encoder_outputs, interm_embeddings, labels_256=labels_256, interact=False)
                if decoder_outputs['valid_mask'] != 1:
                    continue
                ious.append(compute_iou(decoder_outputs['pred_masks'], labels_256))
                bnd_ious.append(compute_boundary_iou(decoder_outputs['pred_masks'], labels_256))
                ious_l.append(compute_iou(decoder_outputs['pred_masks_large'], labels))
                bnd_ious_l.append(compute_boundary_iou(decoder_outputs['pred_masks_large'], labels))
                # ious_pi.append(compute_iou(decoder_outputs['pred_masks_pi'], labels))
                # bnd_ious_pi.append(compute_boundary_iou(decoder_outputs['pred_masks_pi'], labels))
            ious = sum(ious) / len(ious)
            bnd_ious = sum(bnd_ious) / len(bnd_ious)
            ious_l = sum(ious_l) / len(ious_l)
            bnd_ious_l = sum(bnd_ious_l) / len(bnd_ious_l)
            # ious_pi = sum(ious_pi) / len(ious_pi)
            # bnd_ious_pi = sum(bnd_ious_pi) / len(bnd_ious_pi)
            # info_dict = dict(val_iou=ious, bnd_ious=bnd_ious, ious_l=ious_l, bnd_ious_l=bnd_ious_l, ious_pi=ious_pi, bnd_ious_pi=bnd_ious_pi)
            info_dict = {
                f'val_ious_{dataset_name}': ious, 
                f'val_bnd_ious_{dataset_name}': bnd_ious, 
                f'val_ious_l_{dataset_name}': ious_l, 
                f'val_bnd_ious_l_{dataset_name}': bnd_ious_l}
            self._log_info(info_dict, 0, epoch)
            self._write_tensorboard(info_dict, epoch)

    @torch.no_grad()
    def test(self):
        self.sam_encoder.eval()
        self.net.eval()
        for test_dataloaders in self.test_dataloaders:
            datset_name = test_dataloaders.dataset.dataset['data_name'][0]
            print(f"--- test on {datset_name} ---")
            cur_out_dir = os.path.join(self.out_dir, datset_name)
            os.makedirs(cur_out_dir, exist_ok=True)
            for data in tqdm(test_dataloaders):
                inputs, labels = data['image'], data['label']
                
                
                inputs = inputs.to(dist.dev())
                labels = labels.to(dist.dev())
                labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear') 

                # torch.cuda.synchronize()
                # start_time = time.perf_counter()
                batched_input = self.get_batched_input(inputs, labels, input_keys = ['box'])
                encoder_outputs, interm_embeddings = self.sam_encoder(batched_input, multimask_output=False)
                decoder_outputs = self.net(encoder_outputs, interm_embeddings,labels_256=labels_256)

                # torch.cuda.synchronize()
                # if tm == 0:
                #     tm = 1
                # else:
                #     tm += time.perf_counter()-start_time
                #     print(datset_name,cnt,tm-1,'fps',cnt/(tm-1),end='\r')
                # cnt+=1
                #self.save_results(data, decoder_outputs, cur_out_dir)
                #self.save_results(data, decoder_outputs['pred_masks'], cur_out_dir), post_fix='_large.'
                #self.save_results(data, decoder_outputs['pred_masks'], cur_out_dir)
                self.save_results(data, decoder_outputs['pred_masks_large'], cur_out_dir, post_fix='_large.')
                a = 1
            print()

    def save_results(self, data_info, pred_masks, cur_out_dir, post_fix=None):
        bs = len(pred_masks)
        for b_i in range(bs):
            file = data_info['ori_gt_path'][b_i].split('/')[-1]
            if post_fix is not None:
                file_parts = file.split('.')
                file = file_parts[0] + post_fix + file_parts[-1]
            size = (data_info['ori_label'][b_i].shape[1], data_info['ori_label'][b_i].shape[2])
            logit = F.interpolate(pred_masks[b_i].unsqueeze(0), size=size, mode='bilinear')[0, 0]
            # logit = pred_masks[b_i][0]
            mask = (logit > 0).cpu().numpy().astype(np.uint8)
            Image.fromarray(mask * 255).save(os.path.join(cur_out_dir, file))
        pass


@torch.no_grad()
def compute_iou(preds, target):
    # assert target.shape[1] == 1, 'only support one mask per image now'
    # vis_preds(preds, target)
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i],target[i])
    return iou / len(preds)

@torch.no_grad()
def compute_boundary_iou(preds, target):
    # assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i],postprocess_preds[i])
    return iou / len(preds)


def vis_preds(preds, target):
    preds = preds.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    preds = (preds > 0).astype(np.uint8) * 255
    target = (target > 0.5).astype(np.uint8) * 255
    for idx, (p, t) in enumerate(zip(preds, target)):
        p = p[0]
        t = t[0]
        out = np.concatenate([p, t], axis=-1)
        Image.fromarray(out).save(f"tmp/{idx}.png")

    