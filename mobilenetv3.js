// Source:
// https://www.kaggle.com/models/google/mobilenet-v3/frameworks/tfJs/variations/small-100-224-feature-vector/versions/1?tfhub-redirect=true&select=model.json

// Be sure to load TensorFlow.js on your page. See
// https://github.com/tensorflow/tfjs#getting-started.
let asdasd;
let pred;
async function MobileNetV3(){
    // const modelURL = 'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
    const modelURL = 'https://www.kaggle.com/models/google/mobilenet-v3/frameworks/TfJs/variations/small-100-224-feature-vector/versions/1';
    const model = await tf.loadGraphModel(modelURL, {fromTFHub: true});
    console.log('MobileNetv3 loaded');
    model.input = tf.variable(tf.zeros([1, 224, 224, 3]));
    model.output = model.predict(model.input);

    // Warmup the model.
    pred = tf.tidy(() => {
        return model.predict(tf.zeros([1, 224, 224, 3]));
    });

    console.log('MobileNetv3 warmed up');


    asdasd = model;
    return model;
}



