from tensorflow.keras import layers
import tensorflow as tf


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_bn(x, filters, kernel_size,  strides=1, alpha=1, activation=True):
    ilters = _make_divisible(filters * alpha)
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
    )(x)
    x = layers.BatchNormalization(
        momentum=0.9,
        epsilon=0.0010000000474974513,
    )(x)
    if activation:
        x = layers.ReLU(max_value=6)(x)
    return x


def depthwiseConv_bn(x, depth_multiplier, kernel_size,  strides=1):

    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        depth_multiplier=depth_multiplier,
        padding='same',
        use_bias=False,
    )(x)
    x = layers.BatchNormalization(
        momentum=0.9,
        epsilon=0.0010000000474974513,
    )(x)
    x = layers.ReLU(max_value=6)(x)
    return x


def efficientnet_lite(input_shape=(224, 224, 3), alpha=1):
    inputs = layers.Input(shape=input_shape)

    # block 1
    x = conv_bn(inputs, 32, 3,  2, alpha, activation=True)
    x = depthwiseConv_bn(
        x,
        depth_multiplier=1,
        kernel_size=3,
        strides=1
    )
    x = conv_bn(x, 16, 1,  1, alpha, activation=False)

    # block 2
    x = conv_bn(x, 96, 1,  1, alpha, activation=True)
    x = depthwiseConv_bn(
        x,
        depth_multiplier=1,
        kernel_size=3,
        strides=2
    )
    x1 = conv_bn(x, 24, 1,  1, alpha, activation=False)

    # block 3 skip 1
    x = conv_bn(x1, 144, 1,  1, alpha, activation=True)
    x = depthwiseConv_bn(
        x,
        depth_multiplier=1,
        kernel_size=3,
        strides=1
    )
    x = conv_bn(x, 24, 1,  1, alpha, activation=False)
    x = layers.add([x1, x])

    # block 4
    x = conv_bn(x, 144, 1,  1, alpha, activation=True)
    x = depthwiseConv_bn(
        x,
        depth_multiplier=1,
        kernel_size=5,
        strides=2
    )
    x2 = conv_bn(x, 40, 1,  1, alpha, activation=False)

    # block 5 skip 2
    x = conv_bn(x2, 240, 1,  1, alpha, activation=True)
    x = depthwiseConv_bn(
        x,
        depth_multiplier=1,
        kernel_size=5,
        strides=1
    )
    x = conv_bn(x, 40, 1,  1, alpha, activation=False)
    x = layers.add([x2, x])

    # block 6
    x = conv_bn(x, 240, 1,  1, alpha, activation=True)
    x = depthwiseConv_bn(
        x,
        depth_multiplier=1,
        kernel_size=3,
        strides=2
    )
    x3 = conv_bn(x, 80, 1,  1, alpha, activation=False)

    # block 7 skip 3
    x = conv_bn(x3, 480, 1,  1, alpha, activation=True)
    x = depthwiseConv_bn(
        x,
        depth_multiplier=1,
        kernel_size=3,
        strides=1
    )
    x = conv_bn(x, 80, 1,  1, alpha, activation=False)
    x4 = layers.add([x3, x])

    # block 8 skip 4
    x = conv_bn(x4, 480, 1,  1, alpha, activation=True)
    x = depthwiseConv_bn(
        x,
        depth_multiplier=1,
        kernel_size=3,
        strides=1
    )
    x = conv_bn(x, 80, 1,  1, alpha, activation=False)
    x = layers.add([x4, x])

    # # block 9
    x = conv_bn(x, 480, 1,  1, alpha, activation=True)
    x = depthwiseConv_bn(
        x,
        depth_multiplier=1,
        kernel_size=5,
        strides=1
    )
    x5 = conv_bn(x, 112, 1,  1, alpha, activation=False)

    # # block 10 skip 5
    x = conv_bn(x5, 672, 1,  1, alpha, activation=True)
    x = depthwiseConv_bn(
        x,
        depth_multiplier=1,
        kernel_size=5,
        strides=1
    )
    x = conv_bn(x, 112, 1,  1, alpha, activation=False)
    x6 = layers.add([x5, x])

    # # block 10 skip 5
    x = conv_bn(x6, 672, 1,  1, alpha, activation=True)
    x = depthwiseConv_bn(
        x,
        depth_multiplier=1,
        kernel_size=5,
        strides=1
    )
    x = conv_bn(x, 112, 1,  1, alpha, activation=False)
    x = layers.add([x6, x])

    # block 11
    x = conv_bn(x, 672, 1,  1, alpha, activation=True)
    x = depthwiseConv_bn(
        x,
        depth_multiplier=1,
        kernel_size=5,
        strides=2
    )
    x7 = conv_bn(x, 192, 1,  1, alpha, activation=False)

    x = conv_bn(x7, 1152, 1,  1, alpha, activation=True)
    x = depthwiseConv_bn(
        x,
        depth_multiplier=1,
        kernel_size=5,
        strides=1
    )
    x = conv_bn(x, 192, 1,  1, alpha, activation=False)
    x8 = layers.add([x7, x])

    x = conv_bn(x8, 1152, 1,  1, alpha, activation=True)
    x = depthwiseConv_bn(
        x,
        depth_multiplier=1,
        kernel_size=5,
        strides=1
    )
    x = conv_bn(x, 192, 1,  1, alpha, activation=False)
    x9 = layers.add([x8, x])

    x = conv_bn(x9, 1152, 1,  1, alpha, activation=True)
    x = depthwiseConv_bn(
        x,
        depth_multiplier=1,
        kernel_size=5,
        strides=1
    )
    x = conv_bn(x, 192, 1,  1, alpha, activation=False)
    x = layers.add([x9, x])

    x = conv_bn(x, 1152, 1,  1, alpha, activation=True)
    x = depthwiseConv_bn(
        x,
        depth_multiplier=1,
        kernel_size=3,
        strides=1
    )
    x = conv_bn(x, 320, 1,  1, alpha, activation=False)

    x = conv_bn(x, 1280, 1,  1, alpha, activation=True)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1000)(x)

    predictions = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs, predictions)
    return model
