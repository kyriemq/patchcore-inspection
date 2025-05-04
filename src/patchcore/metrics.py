import numpy as np
from sklearn import metrics
from sklearn.metrics import average_precision_score
import scipy.ndimage as ndimage
from tqdm import tqdm

def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )

    image_ap = average_precision_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    return {
        "auroc": auroc,
        "image_ap": image_ap,
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds
    }

def compute_pro_score(anomaly_segmentations, ground_truth_masks, threshold=0.5):
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    num_images = anomaly_segmentations.shape[0]
    pro_scores = []

    for i in range(num_images):
        anomaly_mask = (anomaly_segmentations[i] > threshold).astype(np.float32)
        gt_mask = ground_truth_masks[i].astype(np.float32)

        labeled_mask, num_regions = ndimage.label(gt_mask)
        region_pro_scores = []
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_mask == region_id).astype(np.float32)
            overlap = np.sum(anomaly_mask * region_mask)
            region_area = np.sum(region_mask)
            if region_area == 0:
                continue
            pro_score = overlap / region_area
            region_pro_scores.append(pro_score)

        if region_pro_scores:
            pro_scores.append(np.mean(region_pro_scores))

    return np.mean(pro_scores) if pro_scores else 0.0

def compute_pro(predictions, flat_ground_truth_masks, original_shape):
    num_images = original_shape[0]
    height = original_shape[1]
    width = original_shape[2]
    predictions = predictions.reshape((num_images, height, width))
    flat_ground_truth_masks = flat_ground_truth_masks.reshape((num_images, height, width))
    return compute_pro_score(predictions, flat_ground_truth_masks)

def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    # 确保 flat_ground_truth_masks 是二元标签
    flat_ground_truth_masks_binary = flat_ground_truth_masks.astype(int)
    pixel_ap = average_precision_score(
        flat_ground_truth_masks_binary, flat_anomaly_segmentations
    )

    thresholds = np.linspace(0, 1, 100)
    pros = []
    original_shape = anomaly_segmentations.shape

    for threshold in tqdm(thresholds, desc="Computing PRO for different thresholds"):
        predictions = (flat_anomaly_segmentations >= threshold).astype(int)
        pro = compute_pro(predictions, flat_ground_truth_masks, original_shape)
        pros.append(pro)

    best_threshold = thresholds[np.argmax(pros)]
    best_pro = np.max(pros)

    predictions = (flat_anomaly_segmentations >= best_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "pixel_ap": pixel_ap,
        "pro_score": best_pro,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": best_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim
    }
