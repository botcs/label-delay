function convBlock(inputs, filters, stride) {
    const x = tf.layers.conv2d({
        filters: filters,
        kernelSize: [3, 3],
        strides: [stride, stride],
        padding: 'same',
        useBias: false,
    }).apply(inputs);

    return tf.layers.batchNormalization().apply(x);
}

function bottleneck(inputs, filters, stride, expansion) {
    const x1 = tf.layers.conv2d({
        filters: expansion * inputs.shape[inputs.shape.length - 1],
        kernelSize: [1, 1],
        strides: [1, 1],
        useBias: false
    }).apply(inputs);

    const x2 = convBlock(x1, x1.shape[x1.shape.length - 1], stride);
    const x3 = tf.layers.conv2d({
        filters: filters,
        kernelSize: [1, 1],
        strides: [1, 1],
        useBias: false
    }).apply(x2);

    if (stride === 1 && inputs.shape[inputs.shape.length - 1] === filters) {
        return tf.layers.add().apply([inputs, x3]);
    }

    return x3;
}

function mobileNetV2(inputShape) {
    const inputs = tf.input({shape: inputShape, dataFormat: 'channelsLast'});
    let x = convBlock(inputs, 32, 2);

    // Bottlenecks
    x = bottleneck(x, 16, 1, 1);
    x = bottleneck(x, 24, 2, 6);
    x = bottleneck(x, 24, 1, 6);
    x = bottleneck(x, 32, 2, 6);
    x = bottleneck(x, 32, 1, 6);
    x = bottleneck(x, 32, 1, 6);
    x = bottleneck(x, 64, 2, 6);
    x = bottleneck(x, 64, 1, 6);
    x = bottleneck(x, 64, 1, 6);
    // x = bottleneck(x, 64, 1, 6);
    // x = bottleneck(x, 96, 1, 6);
    // x = bottleneck(x, 96, 1, 6);
    // x = bottleneck(x, 96, 1, 6);
    // x = bottleneck(x, 160, 2, 6);
    // x = bottleneck(x, 160, 1, 6);
    // x = bottleneck(x, 160, 1, 6);
    // x = bottleneck(x, 320, 1, 6);

    // Final layers
    x = convBlock(x, 128, 1);
    x = tf.layers.globalAveragePooling2d("channelsLast").apply(x);
    return tf.model({inputs: inputs, outputs: x});
}