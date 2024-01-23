function convBlock(input, filters, kernelSize = 3, strides = 1, downsample = false) {
    let conv = tf.layers.conv2d({
        filters: filters,
        kernelSize: kernelSize,
        strides: strides,
        padding: 'same',
        activation: 'selu',
        kernelInitializer: 'varianceScaling',
        kernelRegularizer: 'l1l2',
        // useBias: false,
    }).apply(input);
    // conv = tf.layers.batchNormalization().apply(conv);
    if (downsample) {
        conv = tf.layers.maxPooling2d({ 
            poolSize: [2, 2], 
            strides: [2, 2]
        }).apply(conv);
    }
    return conv;
}

function CNN(inputShape, width=64) {
    const input = tf.input({
        shape: inputShape,
        dataFormat: 'channelsLast',
    });
    let x = convBlock(input, width, 7);
    x = convBlock(x, width, 3, 1, true);
    x = convBlock(x, width*2, 3, 1, true);
    x = convBlock(x, width*2, 3, 1, true);
    x = convBlock(x, width*2, 3, 1, true);
    x = tf.layers.globalAveragePooling2d({dataFormat: "channelsLast"}).apply(x);
    x = tf.layers.dense({
        units: 64,
        activation: 'selu',
        kernelInitializer: 'heNormal',
        kernelRegularizer: 'l1l2',
    }).apply(x);
    return tf.model({inputs: input, outputs: x});
}