
import numpy as np
from skimage import measure
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, average_precision_score


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def calculate_pro_score(masks, anomaly_maps, max_steps=200, expect_fpr=0.3):
    labeled_images = np.array(masks)
    labeled_images[labeled_images <= 0.45] = 0
    labeled_images[labeled_images > 0.45] = 1
    labeled_images = labeled_images.astype(bool)

    max_th = anomaly_maps.max()
    min_th = anomaly_maps.min()
    delta = (max_th - min_th) / max_steps

    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(anomaly_maps, dtype=bool)

    for step in range(max_steps):
        thred = max_th - step * delta
        # Segmentation
        binary_score_maps[anomaly_maps <= thred] = 0
        binary_score_maps[anomaly_maps > thred] = 1

        pro = []  # per region overlap
        iou = []  # per image IoU

        for i in range(len(binary_score_maps)):
            # Per region overlap
            label_map = measure.label(labeled_images[i], connectivity=2)
            props = measure.regionprops(label_map)

            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                cropped_mask = prop.filled_image  # Corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)

            # Per image IoU
            intersection = np.logical_and(binary_score_maps[i], labeled_images[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], labeled_images[i]).astype(np.float32).sum()

            if labeled_images[i].any() > 0:
                iou.append(intersection / union)

        ious_mean.append(np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())

        # FPR for pro-auc
        masks_neg = ~labeled_images
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    max_iou = np.max(ious_mean)

    # Default 30% FPR vs pro, pro_auc
    idx = fprs <= expect_fpr
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)
    pros_mean_selected = pros_mean[idx]
    pro_auc_score = auc(fprs_selected, pros_mean_selected)

    return pro_auc_score, max_iou


def is_one_class(gt: np.ndarray):
    gt_ravel = gt.ravel()
    return gt_ravel.sum() == 0 or gt_ravel.sum() == gt_ravel.shape[0]


def calculate_px_metrics(gt_px, pr_px, cal_pro):
    if is_one_class(gt_px):
        return 0, 0, 0, 0, 0

    auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
    precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
    ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())

    if cal_pro:
        aupro, max_iou = calculate_pro_score(gt_px, pr_px)
    else:
        aupro, max_iou = 0, 0

    return auroc_px * 100, f1_px * 100, ap_px * 100, aupro * 100, max_iou * 100


def calculate_im_metrics(gt_im, pr_im):
    if is_one_class(gt_im):
        return 0, 0, 0

    auroc_im = roc_auc_score(gt_im.ravel(), pr_im.ravel())
    precisions, recalls, thresholds = precision_recall_curve(gt_im.ravel(), pr_im.ravel())
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_im = np.max(f1_scores[np.isfinite(f1_scores)])
    ap_im = average_precision_score(gt_im, pr_im)

    return ap_im * 100, auroc_im * 100, f1_im * 100


def calculate_average_metric(metrics: dict):
    average = {}
    for obj, metric in metrics.items():
        for k, v in metric.items():
            if k not in average:
                average[k] = []
            average[k].append(v)

    for k, v in average.items():
        average[k] = np.mean(v)

    return average


def calculate_metric(results, obj, cal_pro=False, by_specie=False, specie=None):
    gt_px = []
    pr_px = []

    gt_im = []
    pr_im = []

    for idx in range(len(results['cls_names'])):
        if results['cls_names'][idx] == obj:
            if (by_specie and results['specie_name'][idx] in [specie, 'good']) or (not by_specie):
                gt_px.append(results['imgs_masks'][idx])
                pr_px.append(results['anomaly_maps'][idx])

                gt_im.append(results['imgs_gts'][idx])
                pr_im.append(results['anomaly_scores'][idx])

    gt_px = np.array(gt_px)
    pr_px = np.array(pr_px)

    gt_im = np.array(gt_im)
    pr_im = np.array(pr_im)

    auroc_px, f1_px, ap_px, aupro, max_iou = calculate_px_metrics(gt_px, pr_px, cal_pro)
    ap_im, auroc_im, f1_im = calculate_im_metrics(gt_im, pr_im)

    metric = {
        'auroc_px': auroc_px,
        'auroc_im': auroc_im,
        'f1_px': f1_px,
        'f1_im': f1_im,
        'ap_px': ap_px,
        'ap_im': ap_im,
        'aupro': aupro,
        'max_iou': max_iou,
    }

    return metric
