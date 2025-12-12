"""
src/losses.py

Loss functions and metrics for segmentation.

For Phase 1 we focus on binary LV-vs-background, so we define:
- dice_coef: metric
- dice_loss: 1 - dice
- bce_dice_loss: BCE + dice_loss (very common in medical segmentation)
"""

import tensorflow as tf
from tensorflow import keras
from config import NUM_CLASSES, SMOOTH


def dice_coef(y_true, y_pred, smooth: float = 1e-6):
    """
    Compute the Dice coefficient for binary segmentation.

    Args:
        y_true: ground truth mask, shape (batch, H, W, 1), values in {0,1}
        y_pred: predicted mask, shape (batch, H, W, 1), values in [0,1]
        smooth: small constant to avoid division by zero.

    Returns:
        Scalar Dice coefficient averaged over the batch.
    """
    # Ensure floating-point tensors
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Flatten spatial dims: (batch, H*W)
    y_true_f = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
    y_pred_f = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1))

    # Intersection and sums per sample
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denom = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)

    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return tf.reduce_mean(dice)


def dice_loss(y_true, y_pred):
    """
    Dice loss = 1 - Dice coefficient.
    """
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """
    Combined Binary Cross-Entropy + Dice loss.

    BCE encourages pixel-wise correctness; Dice encourages good overlap
    and is robust to class imbalance (small LV vs large background).

    This is a strong baseline loss for medical segmentation.
    """
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

def multi_class_dice(y_true, y_pred, exclude_background=True):
    """
    Multi-class Dice coefficient for sparse ground truth masks.

    Args:
        y_true: (B, H, W, 1) sparse integer labels
        y_pred: (B, H, W, C) softmax probabilities
        exclude_background: whether to exclude class 0 from Dice

    Returns:
        Mean Dice coefficient over classes.
    """

    # Remove channel dim: (B, H, W)
    y_true = tf.squeeze(y_true, axis=-1)

    # Convert to one-hot: (B, H, W, C)
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=NUM_CLASSES)

    # Choose which classes to include
    if exclude_background:
        y_true_one_hot = y_true_one_hot[..., 1:]
        y_pred = y_pred[..., 1:]

    # Flatten spatial dims
    y_true_f = tf.reshape(y_true_one_hot, (-1, tf.shape(y_pred)[-1]))
    y_pred_f = tf.reshape(y_pred, (-1, tf.shape(y_pred)[-1]))

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    denominator = tf.reduce_sum(y_true_f + y_pred_f, axis=0)

    dice_per_class = (2.0 * intersection + SMOOTH) / (denominator + SMOOTH)

    return tf.reduce_mean(dice_per_class)

def multi_class_dice_loss(y_true, y_pred):
    return 1.0 - multi_class_dice(y_true, y_pred)

def sparse_cce_multi_class_dice_loss(y_true, y_pred):
    """
    Sparse Categorical Cross Entropy + Multi-Class Dice Loss.
    """

    sce = keras.losses.SparseCategoricalCrossentropy()
    cce_loss = sce(y_true, y_pred)

    dice_loss = multi_class_dice_loss(y_true, y_pred)

    return cce_loss + dice_loss

