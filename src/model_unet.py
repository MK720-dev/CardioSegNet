"""
src/model_unet.py

Baseline U-Net model for LV vs background binary segmentation.

Design choices:
- Input: 128x128, 1 channel (grayscale cardiac MRI)
- 2 downsampling levels + bottleneck (shallow U-Net → fast prototyping)
- BatchNorm after each Conv2D
- ReLU activations
- Output: 1-channel sigmoid (probability of LV at each pixel)
"""

from tensorflow import keras
from tensorflow.keras import layers


def conv_block(x, filters: int, name_prefix: str = "conv"):
    """
    Two Conv2D + BatchNorm + ReLU layers.

    Args:
        x: input tensor
        filters: number of convolution filters
        name_prefix: prefix for layer names (for readability in model summary)

    Returns:
        Output tensor of the block.
    """
    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same",
        use_bias=False,
        name=f"{name_prefix}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu1")(x)

    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same",
        use_bias=False,
        name=f"{name_prefix}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu2")(x)

    return x


def build_unet_baseline(input_shape=(128, 128, 1)):
    """
    Build the baseline U-Net for binary segmentation.

    Architecture:
    - Encoder:
        Level 1: 32 filters, MaxPool
        Level 2: 64 filters, MaxPool
    - Bottleneck:
        128 filters
    - Decoder:
        Up to 64, then 32 filters with skip connections
    - Output:
        1x1 Conv2D (1 filter) + sigmoid

    Args:
        input_shape: (H, W, C), default (128, 128, 1).

    Returns:
        Keras Model.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Encoder
    c1 = conv_block(inputs, 32, name_prefix="enc1")
    p1 = layers.MaxPool2D((2, 2), name="pool1")(c1)  # -> 64x64

    c2 = conv_block(p1, 64, name_prefix="enc2")
    p2 = layers.MaxPool2D((2, 2), name="pool2")(c2)  # -> 32x32

    # Bottleneck
    bn = conv_block(p2, 128, name_prefix="bottleneck")

    # Decoder
    u2 = layers.UpSampling2D((2, 2), name="up2")(bn)  # -> 64x64
    u2 = layers.Concatenate(name="skip2")([u2, c2])
    c3 = conv_block(u2, 64, name_prefix="dec2")

    u1 = layers.UpSampling2D((2, 2), name="up1")(c3)  # -> 128x128
    u1 = layers.Concatenate(name="skip1")([u1, c1])
    c4 = conv_block(u1, 32, name_prefix="dec1")

    # Output layer: 1-channel sigmoid for LV probability
    outputs = layers.Conv2D(
        1,
        (1, 1),
        activation="sigmoid",
        name="output_mask",
    )(c4)

    model = keras.Model(inputs, outputs, name="unet_baseline_lv_128")
    return model

def build_unet_deep(input_shape=(256, 256, 1)):
    """
    Build the deep U-Net model for multi-class segmentation (LV, RV and MYO).

    Architecture:
    - Encoder:
        Level 1: 32 filters, MaxPool
        Level 2: 64 filters, MaxPool
        Level 3:  128 filters, MaxPool
        Level 4: 256 filters, MaxPool
    - Bottleneck:
        512 filters
    - Decoder:
        Up to 256, 128, 64 then 32 filters with skip connections
    - Output:
        1x1 Conv2D (4-channels filter) + softmax

    Args:
        input_shape: (H, W, C), default (256, 256, 1).

    Returns:
        Keras Model.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Encoder 
    c1 = conv_block(inputs, 32, name_prefix="enc1")
    p1 = layers.MaxPool2D((2, 2), name="pool1")(c1)  # -> 128x128

    c2 = conv_block(p1, 64, name_prefix="enc2")
    p2 = layers.MaxPool2D((2, 2), name="pool2")(c2)  # -> 64x64

    c3 = conv_block(p2, 128, name_prefix="enc3")
    p3 = layers.MaxPool2D((2,2), name="pool3")(c3) # -> 32x32

    c4 = conv_block(p3, 256, name_prefix="enc4")
    p4 = layers.MaxPool2D((2,2), name="pool4")(c4) # -> 16*16

    # Bottleneck 
    bn = conv_block(p4, 512, name_prefix="bottleneck")

    # Decoder
    u4 = layers.UpSampling2D((2, 2), name="up4")(bn)  # -> 32*32
    u4 = layers.Concatenate(name="skip4")([u4, c4])
    c5 = conv_block(u4, 256, name_prefix="dec4")

    u3 = layers.UpSampling2D((2, 2), name="up3")(c5)  # -> 64*64
    u3 = layers.Concatenate(name="skip3")([u3, c3])
    c6 = conv_block(u3, 128, name_prefix="dec3")

    u2 = layers.UpSampling2D((2, 2), name="up2")(c6)  # -> 128*128
    u2 = layers.Concatenate(name="skip2")([u2, c2])
    c7 = conv_block(u2, 64, name_prefix="dec2")

    u1 = layers.UpSampling2D((2, 2), name="up1")(c7)  # -> 256*256
    u1 = layers.Concatenate(name="skip1")([u1, c1])
    c8 = conv_block(u1, 32, name_prefix="dec1")

    # Output layer: 4-channel softmax for Background, LV, RV and MYO probability
    outputs = layers.Conv2D(
        4,
        (1, 1),
        activation="softmax",
        name="output_mask",
    )(c8)

    model = keras.Model(inputs, outputs, name="unet_deep_256")

    return model