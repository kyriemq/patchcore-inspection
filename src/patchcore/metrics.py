import numpy as np
from sklearn.metrics import average_precision_score

def compute_image_ap(anomaly_prediction_weights, anomaly_ground_truth_labels):
    """
    Computes the average precision (AP) for image-wise anomaly detection.

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    return average_precision_score(anomaly_ground_truth_labels, anomaly_prediction_weights)

def compute_pixel_ap(anomaly_segmentations, ground_truth_masks):
    """
    Computes the average precision (AP) for pixel-wise anomaly detection.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    return average_precision_score(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)

def compute_pro(anomaly_segmentations, ground_truth_masks):
    """
    Computes the PRO score. This is a placeholder function and needs to be implemented.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    # Placeholder implementation, replace with actual PRO calculation
    return np.random.rand()
