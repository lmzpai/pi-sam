import multiprocessing as mp
import os

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import gradio as gr
import gradio_image_prompter as gr_ext

from segment_anything import sam_model_registry
from segment_anything.utils import misc,dist
from segment_anything import sam_model_registry, build_mask_decoder



net = None
sam = None
ori_img = None


class ARGS():
    def __init__(self):
        self.output=""
        self.model_type="vit_h"
        self.checkpoint=""
        self.device="cuda"
        self.restore_model=""
        self.gpu = 0
        self.decoder_type="pi"
        self.decoder_ckpt=""
        self.world_size=1
        self.dist_url='env://'
        self.rank = 0
        self.local_rank = 0
        self.find_unused_params = True




net = None
sam = None
ori_img = None

def net_init():
    global net,sam
    args = ARGS()
    net = build_mask_decoder(args.decoder_type, args.model_type, args.decoder_ckpt).to(dist.dev())
    net_without_ddp = net
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam = sam.to(dist.dev())
    sam.eval()
    if args.restore_model:
        print("restore model from:", args.restore_model)
        net_without_ddp.load_state_dict(torch.load(args.restore_model,map_location="cpu")['model_state_dict'])
        net_without_ddp.eval()
    net.eval()


def interactivate_sam(click_img,ori_img,gt):
    img, fg_points, bg_points = None,[],[]
    if click_img is not None:
        img, points = click_img["image"], click_img["points"]
        w,h=img.shape[0],img.shape[1]
        points = np.array(points).reshape((-1, 2, 3))
        points = points.reshape((-1, 3))
        fg_points = points[np.where(points[:, 2] == 1)[0]][:, :2].copy()
        bg_points = points[np.where(points[:, 2] == 0)[0]][:, :2].copy()
        lt = points[np.where(points[:, 2] == 2)[0]][None, :, :]
        rb = points[np.where(points[:, 2] == 3)[0]][None, :, :]
        poly = points[np.where(points[:, 2] <= 1)[0]][None, :, :]
        points = [lt, rb, poly] if len(lt) > 0 else [poly, np.array([[[0, 0, 4]]])]
        points = np.concatenate(points, axis=1)
        fg_points[:, 0] /= h #y
        fg_points[:, 1] /= w #x
        bg_points[:, 0] /= h
        bg_points[:, 1] /= w
        ##这里是先列后行
        fg_points = torch.tensor(fg_points,device = 'cuda').unsqueeze(0)
        bg_points = torch.tensor(bg_points,device = 'cuda').unsqueeze(0)

    labels_val = F.interpolate(torch.unsqueeze(torch.from_numpy(gt).float()[...,0],0).unsqueeze(0),[1024,1024],mode='bilinear')
    labels_box = misc.masks_to_boxes(labels_val[:,0,:,:])
    imgs = F.interpolate(torch.unsqueeze(torch.from_numpy(ori_img).float(),0).permute(0,3,1,2),[1024,1024],mode='bilinear').permute(0, 2, 3, 1).cpu().numpy()
    dict_input = dict()
    dict_input['image'] = torch.as_tensor(imgs.astype(dtype=np.uint8)[0], device=sam.device).permute(2, 0, 1).contiguous()
    dict_input['original_size'] = [1024,1024]
    dict_input['point_coords'] = torch.cat([fg_points, bg_points], dim=1).to(sam.device)
    dict_input['point_labels'] = torch.cat([torch.ones(fg_points.shape[1]), torch.zeros(bg_points.shape[1])], dim=0).unsqueeze(0).to(sam.device)
    dict_input['boxes'] = labels_box.to(sam.device)
    batched_input=[dict_input]
    with torch.no_grad():
        encoder_outputs, interm_embeddings = sam(batched_input, multimask_output=False)
        decoder_outputs = net(encoder_outputs, interm_embeddings, labels_256=None, fg=fg_points, bg=bg_points, interactive=True)

    masks_pi = decoder_outputs['pred_masks_large'].detach()
    masks_pi_vis = F.interpolate(masks_pi,img.shape[:2],mode='bilinear')
    masks_pi_vis = (masks_pi_vis > 0).cpu().numpy().astype(np.uint8)[0,0]
    cv2.imwrite("mask.png",masks_pi_vis*255)

    img_plus_gt = ori_img.copy().astype(np.uint16)
    img_plus_gt[np.where(masks_pi_vis==1)]+=np.array([255,0,255]).astype(np.uint16)
    img_plus_gt[np.where(masks_pi_vis==1)]//=2
    img_plus_gt = img_plus_gt.astype(np.uint8)

    img_gt = np.zeros_like(ori_img)
    img_gt[np.where(masks_pi_vis==1)]=(255,255,255)
    return {"image":img_plus_gt},(ori_img, [[masks_pi_vis,'mask']])  

def get_click_examples():
    assets_dir = 'DIS5K/DIS-TE4/im/'
    gt_dir = 'DIS5K/DIS-TE4/gt/'
    app_images = list(os.listdir(assets_dir))
    app_gt = list(os.listdir(gt_dir))
    ret = [[{"image": os.path.join(assets_dir, x)},os.path.join(gt_dir, y),os.path.join(assets_dir, x)] for x,y in zip(app_images,app_gt)]
    return ret


def on_submit_btn(click_img,ori_img,gt):
    return interactivate_sam(click_img,ori_img,gt)

class ServingCommand(object):
    """Command to run serving."""

    def __init__(self, output_queue):
        self.output_queue = output_queue
        self.output_dict = mp.Manager().dict()
        self.output_index = mp.Value("i", 0)

    def postprocess_outputs(self, outputs):
        """Main the detection objects."""
        scores, masks = outputs["scores"], outputs["masks"]
        concepts, captions = outputs["concepts"], outputs["captions"]
        text_template = "{} ({:.2f}, {:.2f}): {}"
        text_contents = concepts, scores[:, 0], scores[:, 1], captions
        texts = np.array([text_template.format(*vals) for vals in zip(*text_contents)])
        return masks, texts

    def run(self):
        """Main loop to make the serving outputs."""
        while True:
            img_id, outputs = self.output_queue.get()
            self.output_dict[img_id] = self.postprocess_outputs(outputs)

def build_gradio_app():
    """Build the gradio application."""

    title = "Precise Interactive SAM"
    header = (
        "<div align='center'>"
        "<h1>Precise Interactive SAM</h1>"
        "<h3></h3>"  # noqa
        "</div>"
    )
    theme = "soft"
    js = """document.addEventListener('contextmenu',function(e){
    e.preventDefault();
    })"""
    css = """#anno-img .mask {opacity: 0.5; transition: all 0.2s ease-in-out;}
             #anno-img .mask.active {opacity: 0.7}"""
    
    def on_reset_btn():
        click_img, draw_img = gr.Image(None), gr.ImageEditor(None)
        anno_img = gr.AnnotatedImage(None)
        return click_img, draw_img, anno_img
    
    app, _ = gr.Blocks(title=title, theme=theme, css=css,js=js).__enter__(), gr.Markdown(header)
    container, column = gr.Row().__enter__(), gr.Column().__enter__()
    click_tab, click_img, gt, inv_pic = gr.Tab("Point+Box").__enter__(), gr_ext.ImagePrompter(show_label=False), gr.Image(show_label=False), gr.Image(visible=False)
    interactions = "LeftClick (FG) | MiddleClick (BG) | PressMove (Box)"
    gr.Markdown("<h3 style='text-align: center'>[🖱️ | 🖐️]: 🌟🌟 {} 🌟🌟 </h3>".format(interactions))
    gr.Examples(get_click_examples(),inputs=[click_img,gt,inv_pic])
    _, draw_tab = click_tab.__exit__(), gr.Tab("Sketch").__enter__()
    draw_img, _ = gr.ImageEditor(show_label=False), draw_tab.__exit__()
    _,  column = column.__exit__(), gr.Column().__enter__()
    row, reset_btn, submit_btn = gr.Row().__enter__(), gr.Button("Reset"), gr.Button("Execute")
    row.__exit__()
    anno_img = gr.AnnotatedImage(elem_id="anno-img", show_label=False)
    reset_btn.click(on_reset_btn, [], [click_img, draw_img, anno_img])
    submit_btn.click(on_submit_btn, [click_img,inv_pic,gt], [click_img,anno_img])
    column.__exit__(), container.__exit__(), app.__exit__()
    return app

if __name__ == "__main__":
    os.environ["LOCAL_RANK"] = "0"
    net_init()
    app = build_gradio_app()
    app.queue()
    app.launch()