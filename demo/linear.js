function Linear(inputShape, width=64) {
    const input = tf.input({shape: inputShape, dataFormat: 'channelsLast'});
    let x = tf.layers.flatten().apply(input);
    return tf.model({inputs: input, outputs: x});
}
