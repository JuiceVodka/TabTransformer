def f1_from_tokens_rough(gt_tokens, pred_tokens):
    """
    Computes F1 score between two token sequences (as sets).
    Args:
        gt_tokens (list or np.ndarray): Ground truth token sequence.
        pred_tokens (list or np.ndarray): Predicted token sequence.
    Returns:
        precision, recall, f1 (float): F1 metrics.
    """
    gt_set = set(gt_tokens)
    pred_set = set(pred_tokens)
    true_positives = len(gt_set & pred_set)
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(gt_set) if gt_set else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def f1_from_tokens_strict(gt_tokens, pred_tokens):
    """
    Computes F1 score between two token sequences, considering duplicates but not order
    Args:
        gt_tokens (list or np.ndarray): Ground truth token sequence.
        pred_tokens (list or np.ndarray): Predicted token sequence.
    Returns:
        precision, recall, f1 (float): F1 metrics.
    """
    from collections import Counter

    gt_counter = Counter(gt_tokens)
    pred_counter = Counter(pred_tokens)
    true_positives = sum((gt_counter & pred_counter).values())
    precision = true_positives / len(pred_tokens) if pred_tokens else 0.0
    recall = true_positives / len(gt_tokens) if gt_tokens else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1