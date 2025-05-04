"""Anomaly metrics."""
import numpy as np
from sklearn import metrics
from sklearn.metrics import average_precision_score

def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    # 新增图像 AP 计算
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

def compute_pro_score(anomaly_segmentations, ground_truth_masks):
    """
    Compute PRO score (Per-Region Overlap) for anomaly segmentation.
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    pro_scores = []
    for seg, mask in zip(anomaly_segmentations, ground_truth_masks):
        mask = mask.astype(np.bool_)
        seg = (seg > 0.5).astype(np.bool_)  # 假设分割结果已归一化到 [0,1]
        
        # 计算每个区域的 PRO
        labeled_mask, num_regions = ndimage.label(mask)
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_mask == region_id)
            overlap = np.logical_and(region_mask, seg)
            region_area = np.sum(region_mask)
            if region_area == 0:
                continue
            pro_score = np.sum(overlap) / region_area
            pro_scores.append(pro_score)
    
    return np.mean(pro_scores) if pro_scores else 0.0


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel().astype(np.int32)

    # 计算像素级 AUROC
    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks, flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks, flat_anomaly_segmentations
    )

    # 计算像素级 AP
    pixel_ap = average_precision_score(
        flat_ground_truth_masks, flat_anomaly_segmentations
    )

    # 计算 PRO 指标
    pro_score = compute_pro_score(anomaly_segmentations, ground_truth_masks)

    # 其他原有逻辑（如最优阈值）保持不变
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
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(np.int32)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "pixel_ap": pixel_ap,  # 新增
        "pro_score": pro_score,  # 新增
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim
    }
