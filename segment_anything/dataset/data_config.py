import os


dataset_dis = {
    "name": "DIS5K-TR",
    "im_dir": "DIS5K/DIS-TR/im",
    "gt_dir": "DIS5K/DIS-TR/gt",
    "im_ext": ".jpg",
    "gt_ext": ".png"}

dataset_thin = {
    "name": "ThinObject5k-TR",
    "im_dir": "ThinObject5K/images_train",
    "gt_dir": "ThinObject5K/masks_train",
    "im_ext": ".jpg",
    "gt_ext": ".png"}

dataset_hrsod = {
    "name": "HRSOD-TR",
    "im_dir": "HRSOD_release/HRSOD_train",
    "gt_dir": "HRSOD_release/HRSOD_train_mask",
    "im_ext": ".jpg",
    "gt_ext": ".png"}

dataset_uhrsd = {
    "name": "UHRSD-TR",
    "im_dir": "UHRSD_TR_2K/image",
    "gt_dir": "UHRSD_TR_2K/mask",
    "im_ext": ".jpg",
    "gt_ext": ".png"}

dataset_fss = {
    "name": "FSS",
    "im_dir": "fss_all",
    "gt_dir": "fss_all",
    "im_ext": ".jpg",
    "gt_ext": ".png"}

dataset_duts = {
    "name": "DUTS-TR",
    "im_dir": "DUTS-TR",
    "gt_dir": "DUTS-TR",
    "im_ext": ".jpg",
    "gt_ext": ".png"}

dataset_duts_te = {
    "name": "DUTS-TE",
    "im_dir": "DUTS-TE",
    "gt_dir": "DUTS-TE",
    "im_ext": ".jpg",
    "gt_ext": ".png"}

dataset_ecssd = {
    "name": "ECSSD",
    "im_dir": "ecssd",
    "gt_dir": "ecssd",
    "im_ext": ".jpg",
    "gt_ext": ".png"}

dataset_msra = {
    "name": "MSRA10K",
    "im_dir": "MSRA_10K",
    "gt_dir": "MSRA_10K",
    "im_ext": ".jpg",
    "gt_ext": ".png"}

# test set
dataset_dis_te1 = {
    "name": "DIS5K-TE1",
    "im_dir": "DIS5K/DIS-TE1/im",
    "gt_dir": "DIS5K/DIS-TE1/gt",
    "im_ext": ".jpg",
    "gt_ext": ".png"}
dataset_dis_te2 = {
    "name": "DIS5K-TE2",
    "im_dir": "DIS5K/DIS-TE2/im",
    "gt_dir": "DIS5K/DIS-TE2/gt",
    "im_ext": ".jpg",
    "gt_ext": ".png"}
dataset_dis_te3 = {
    "name": "DIS5K-TE3",
    "im_dir": "DIS5K/DIS-TE3/im",
    "gt_dir": "DIS5K/DIS-TE3/gt",
    "im_ext": ".jpg",
    "gt_ext": ".png"}
dataset_dis_te4 = {
    "name": "DIS5K-TE4",
    "im_dir": "DIS5K/DIS-TE4/im",
    "gt_dir": "DIS5K/DIS-TE4/gt",
    "im_ext": ".jpg",
    "gt_ext": ".png"}
dataset_coift_val = {
    "name": "COIFT",
    "im_dir": "COIFT/images",
    "gt_dir": "COIFT/masks",
    "im_ext": ".jpg",
    "gt_ext": ".png"}

dataset_hrsod_val = {
    "name": "HRSOD-TE",
    "im_dir": "HRSOD_release/HRSOD_test",
    "gt_dir": "HRSOD_release/HRSOD_test_mask",
    "im_ext": ".jpg",
    "gt_ext": ".png"}

dataset_thin_val = {
    "name": "ThinObject5k-TE",
    "im_dir": "ThinObject5K/images_test",
    "gt_dir": "ThinObject5K/masks_test",
    "im_ext": ".jpg",
    "gt_ext": ".png"}

dataset_dis_val = {
    "name": "DIS5K-VD",
    "im_dir": "DIS5K/DIS-VD/im",
    "gt_dir": "DIS5K/DIS-VD/gt",
    "im_ext": ".jpg",
    "gt_ext": ".png"}

dataset_uhrsd_val = {
    "name": "UHRSD-TE",
    "im_dir": "UHRSD_TE_2K/image",
    "gt_dir": "UHRSD_TE_2K/mask",
    "im_ext": ".jpg",
    "gt_ext": ".png"}

dataset_david_val ={
    "name": "DAVID-TE",
    "im_dir": "DAVIDS/image",
    "gt_dir": "DAVIDS/gt",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}
dataset_big_test = {
    "name": "BIG_test",
    "im_dir": "BIG/test/img",
    "gt_dir": "BIG/test/gt",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}
dataset_big_val = {
    "name": "BIG_val",
    "im_dir": "BIG/val/img",
    "gt_dir": "BIG/val/gt",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}
# "HRS10K","HRS10K_HD"_2560max
dataset_hrs_val = {
    "name": "HRS10K",
    "im_dir": "HRS10K/img_test",
    "gt_dir": "HRS10K/label_test",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}
dataset_hrshd_val = {
    "name": "HRS10K_HD",
    "im_dir": "HRS10K/img_test_2560max",
    "gt_dir": "HRS10K/label_test_2560max",
    "im_ext": ".jpg",
    "gt_ext": ".png"
}
dataset_voc_val = {
    "name": "VOC",
    "im_dir": "VOC2012/new",
    "gt_dir": "VOC2012/new_gt",
    "im_ext": ".png",
    "gt_ext": ".png"
}
all_train_datasets = [dataset_dis, dataset_thin, dataset_hrsod, dataset_fss, dataset_duts, dataset_duts_te, dataset_ecssd, dataset_msra, dataset_uhrsd]
all_valid_datasets = [dataset_dis_te1, dataset_dis_te2, dataset_dis_te3, dataset_dis_te4, dataset_dis_val, dataset_thin_val, dataset_hrsod_val, dataset_uhrsd_val]

def collect_datasets(used_datasets, data_root, training):
    datasets = []
    if training:
        for dataset in all_train_datasets:
            if dataset['name'] in used_datasets or 'ALL' in used_datasets:
                dataset['im_dir'] = os.path.join(data_root, dataset['im_dir'])
                dataset['gt_dir'] = os.path.join(data_root, dataset['gt_dir'])
                datasets.append(dataset)
    else:
        for dataset in all_valid_datasets:
            if dataset['name'] in used_datasets or 'ALL' in used_datasets:
                dataset['im_dir'] = os.path.join(data_root, dataset['im_dir'])
                dataset['gt_dir'] = os.path.join(data_root, dataset['gt_dir'])
                datasets.append(dataset)
    return datasets