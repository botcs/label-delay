class customModelConnector {
    constructor(kwargs) {
        const {
            architecture,
            image_size,
            image_channels,
            num_classes,
            num_features,
            optimizer
        } = kwargs;
        this.architecture = architecture;
        this.image_size = image_size;
        this.image_channels = image_channels;
        this.num_classes = num_classes;
        this.num_features = num_features;
        this.optimizer = optimizer;
        this.model = null;
    }
    async loadModel() {
        const backboneFactory = {
            "cnn_base": CNN_BASE,
            "cnn_small": CNN_SMALL,
            "cnn_large": CNN_LARGE,
            "resnet18": resNet18,
        }[this.architecture];
        const backbone = await backboneFactory([
            this.image_size,
            this.image_size,
            this.image_channels
        ]);
        this.backbone = backbone;
        const backboneOutputShape = backbone.output.shape.slice(1);
        
        const input = tf.input({shape: backboneOutputShape});
        const proj = tf.layers.dense({
            units: this.num_features,
            activation: "tanh",
            kernelInitializer: "varianceScaling",
            kernelRegularizer: "l1l2",
        }).apply(input);
        const logit = tf.layers.dense({units: this.num_classes}).apply(proj);
        
        this.heads = tf.model({
            inputs: input, 
            outputs: [logit, proj]
        });
    
        // Warm up the model by training it once
        const warmupData = tf.randomNormal([6].concat(backboneOutputShape));
        const labels = tf.oneHot(tf.tensor1d([0,0,1,1,2,2], 'int32'), this.num_classes);
        let logits;
        let loss;
        await OPTIMIZER.minimize(() => {
            logits = this.heads.predict(warmupData)[0];
            loss = tf.losses.softmaxCrossEntropy(labels, logits).mul(0);
            return loss;
        });

        // clean up the tensors
        warmupData.dispose();
        logits.dispose();
        labels.dispose();
        loss.dispose();

        // Summary
        console.log("Backbone:");
        this.backbone.summary();

        console.log("Heads:");
        this.heads.summary();
    }

    forwardBackbone(imageData) {
        return this.backbone.predict(imageData);
    }

    forwardHead(feature) {
        return this.heads.predict(feature);
    }

    predict(imageData) {
        const backboneFeature = this.forwardBackbone(imageData);
        return this.forwardHead(backboneFeature);
    }

}

class TFHubModelConnector {
    constructor(kwargs){
        const {
            architecture,
            image_size,
            image_channels,
            num_classes,
            num_features,
            optimizer
        } = kwargs;
        this.architecture = architecture;
        this.image_size = image_size;
        this.image_channels = image_channels;
        this.num_classes = num_classes;
        this.num_features = num_features;
        this.optimizer = optimizer;
    }

    async loadModel() {
        const modelURL = {
            // "mobilenetv2": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
            // "mobilenetv3": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5"
            "mobilenetv3": "https://www.kaggle.com/models/google/mobilenet-v3/frameworks/TfJs/variations/small-100-224-feature-vector/versions/1"
        }[this.architecture];
        const backbone = await tf.loadGraphModel(modelURL, {fromTFHub: true});
        this.backbone = backbone;

        this.featureInput = tf.input({shape: [1024]});
        this.proj = tf.layers.dense({
            units: this.num_features,
            activation: "selu",
        }).apply(this.featureInput);

        this.logit = tf.layers.dense({
            units: this.num_classes
        }).apply(this.proj);

        this.heads = tf.model({
            inputs: this.featureInput,
            outputs: [this.logit, this.proj]
        });
        
    }

    preprocess(imageTensor) {
        // Assuming imageTensor has shape [batchSize, height, width, channels]
        const batchSize = imageTensor.shape[0];
        const height = imageTensor.shape[1];
        const width = imageTensor.shape[2];
        const squareCrop = [];
    
        for (let i = 0; i < batchSize; i++) {
            const widthToHeight = width / height;
            if (widthToHeight > 1) {
                const heightToWidth = height / width;
                const cropTop = (1 - heightToWidth) / 2;
                const cropBottom = 1 - cropTop;
                squareCrop.push([cropTop, 0, cropBottom, 1]);
            } else {
                const cropLeft = (1 - widthToHeight) / 2;
                const cropRight = 1 - cropLeft;
                squareCrop.push([0, cropLeft, 1, cropRight]);
            }
        }
    
        // Generate an array of indices for the batch, [0, 1, ..., batchSize - 1]
        const boxInd = Array.from({length: batchSize}, (v, k) => k);
    
        // Crop and resize each image in the batch
        const crop = tf.image.cropAndResize(imageTensor, squareCrop, boxInd, [224, 224]);
    
        // Normalize pixel values
        return crop.div(255);
    }
    

    forwardBackbone(imageData) {
        imageData = this.preprocess(imageData);
        return this.backbone.predict(imageData);
    }

    forwardHead(feature) {
        return this.heads.predict(feature);
    }

    predict(imageData) {
        const backboneFeature = this.forwardBackbone(imageData);
        return this.forwardHead(backboneFeature);

    }
}

class GenericModelConnector {
    constructor(kwargs){
        const {
            architecture,
            image_size,
            image_channels,
            num_classes,
            num_features,
            optimizer
        } = kwargs;
        this.architecture = architecture;
        this.image_size = image_size;
        this.image_channels = image_channels;
        this.num_classes = num_classes;
        this.num_features = num_features;
        this.optimizer = optimizer;
        this.model = null;
    }

    async loadModel() {
        let connector;
        if (this.architecture.includes("mobilenet")) {
            connector = TFHubModelConnector;
        } else {
            connector = customModelConnector;
        }
        const model = new connector({
            architecture: this.architecture,
            image_size: this.image_size,
            image_channels: this.image_channels,
            num_classes: this.num_classes,
            num_features: this.num_features,
            optimizer: this.optimizer
        });
        await model.loadModel();
        console.log(model);
        this.model = model;
    }

    forwardBackbone(imageData) {
        return this.model.forwardBackbone(imageData);
    }

    forwardHead(feature) {
        return this.model.forwardHead(feature);
    }

    predict(imageData) {
        const ret = this.model.predict(imageData);
        return ret;
    }
}