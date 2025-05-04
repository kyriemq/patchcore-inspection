import numpy as np
from sklearn import metrics
from sklearn.metrics import average_precision_score
import scipy.ndimage as ndimage  

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
        "image_ap": image_ap,  # 新增
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

def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks, anomaly_ratio=0.1):
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel().astype(np.int32)

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks, flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks, flat_anomaly_segmentations
    )

    pixel_ap = average_precision_score(
        flat_ground_truth_masks, flat_anomaly_segmentations
    )

    # 自适应阈值选择
    threshold = np.quantile(flat_anomaly_segmentations, 1 - anomaly_ratio)
    predictions = (flat_anomaly_segmentations >= threshold).astype(int)

    pro_score = compute_pro_score(predictions.reshape(anomaly_segmentations.shape), ground_truth_masks, threshold=0.5)

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks, flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions_optimal = (flat_anomaly_segmentations >= optimal_threshold).astype(np.int32)
    fpr_optim = np.mean(predictions_optimal > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions_optimal < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "pixel_ap": pixel_ap, 
        "pro_score": pro_score,  
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
        "adaptive_threshold": threshold
    }
