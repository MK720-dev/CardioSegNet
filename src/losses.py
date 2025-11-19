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

