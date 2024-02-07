function convBlock(input, filters, kernelSize = 3, strides = 1) {
    const conv = tf.layers.conv2d({
        filters: filters,
        kernelSize: kernelSize,
        strides: strides,
        padding: 'same',
        activation: 'selu',
        kernelInitializer: 'heNormal'
    }).apply(input);

    return tf.layers.batchNormalization().apply(conv);
}

function residualBlock(input, filters, downsample = false) {
    const strides = downsample ? 2 : 1;
    let conv1 = convBlock(input, filters, 3, strides);
    let conv2 = convBlock(conv1, filters);

    let shortcut = input;
    if (downsample) {
        shortcut = tf.layers.conv2d({
            filters: filters,
            kernelSize: 1,
            strides: 2,
            padding: 'same',
            kernelInitializer: 'heNormal'
        }).apply(input);
        shortcut = tf.layers.batchNormalization().apply(shortcut);
    }

    return tf.layers.add().apply([conv2, shortcut]);
}

function resNet18(inputShape, width=64) {
    const input = tf.input({shape: inputShape, dataFormat: 'channelsLast'});
    let x = convBlock(input, width, 7, 2);
    x = tf.layers.maxPooling2d({ poolSize: [3, 3], strides: [2, 2], padding: 'same' }).apply(x);

    x = residualBlock(x, width);
    x = residualBlock(x, width);

    x = residualBlock(x, width*2, true);
    x = residualBlock(x, width*2);

    x = residualBlock(x, width*4, true);
    x = residualBlock(x, width*4);

    x = residualBlock(x, width*8, true);
    x = residualBlock(x, width*8);

    x = tf.layers.globalAveragePooling2d({dataFormat: "channelsLast"}).apply(x);

    return tf.model({ inputs: input, outputs: x });
}
