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
            "linear": Linear,
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

    getWeights() {
        return this.heads.getWeights();
    }

    setWeights(weights) {
        this.heads.setWeights(weights);
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
            "mobilenetv2": "https://www.kaggle.com/models/google/mobilenet-v2/frameworks/TfJs/variations/140-224-feature-vector/versions/3",
            "mobilenetv3": "https://www.kaggle.com/models/google/mobilenet-v3/frameworks/TfJs/variations/small-100-224-feature-vector/versions/1"
        }[this.architecture];
        const backbone = await tf.loadGraphModel(modelURL, {fromTFHub: true});
        this.backbone = backbone;

        const outputDim = {
            "mobilenetv2": 1792,
            "mobilenetv3": 1024,
        }[this.architecture];

        this.featureInput = tf.input({shape: [outputDim]});

        // layer normalization
        // const x = tf.layers.layerNormalization().apply(this.featureInput);
        const x = this.featureInput;

        this.proj = tf.layers.dense({
            units: this.num_features,
            // activation: "none",
        }).apply(x);

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

        return crop;
        // Normalize pixel values
        // return crop.div(255);
    }
    
    getWeights() {
        return this.heads.getWeights();
    }

    setWeights(weights) {
        this.heads.setWeights(weights);
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

    getWeights() {
        return this.model.getWeights();
    }

    setWeights(weights) {
        this.model.setWeights(weights);
    }
}

class VisLogger {
    // A class for logging data to the Visor in real time
    constructor({
        name = "Log",
        tab = "History",
        xLabel = "Iteration",
        yLabel = "Y",
        height = 300,
    }) {
        tfvis.visor().close();

        this.numUpdates = 0;
        this.X = [];
        this.Y = [];
        this.yLabel = yLabel;
        this.surface = tfvis.visor().surface({ name: name, tab: tab });
        this.axisSettings = { xLabel: xLabel, yLabel: yLabel, height: height };

        this.lastUpdateTime = 0;
        this.timeoutId = null;

        // Create a canvas element for Chart.js
        this.canvas = document.createElement('canvas');
        this.surface.drawArea.appendChild(this.canvas);

        this.chart = new Chart(this.canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: yLabel,
                    data: [],
                    // borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                    // fill: false
                }]
            },
            options: {
                responsive: true,
                animation: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: xLabel
                        },
                        ticks: {
                            animation: false // Disable animation for x-axis ticks
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: yLabel
                        },
                        ticks: {
                            animation: false // Disable animation for y-axis ticks
                        }
                    }
                }
            }
        });
        this.chart.data.labels = this.X;
        this.chart.data.datasets[0].data = this.Y;
    }

    push(data) {
        var x, y;
        if (typeof data === "number") {
            x = this.numUpdates;
            y = data;
        } else {
            x = data.x;
            y = data.y;
        }
        this.X.push(x);
        this.Y.push(y);

        if (this.X.length > 150) {
            // Subsample the data if it gets too long
            this.X = this.X.filter((_, i) => i % 2 === 0);
            this.Y = this.Y.filter((_, i) => i % 2 === 0);
            
            this.chart.data.labels = this.X;
            this.chart.data.datasets[0].data = this.Y;
        }

        this.chart.update();
        this.numUpdates++;
    }

    scheduleUpdate() {
        const now = Date.now();
        const timeSinceLastUpdate = now - this.lastUpdateTime;

        if (timeSinceLastUpdate >= this.updateInterval) {
            this.updateVisualization();
        } else {
            if (this.timeoutId) {
                clearTimeout(this.timeoutId); // Cancel the existing timeout
            }
            this.timeoutId = setTimeout(() => {
                this.updateVisualization();
            }, this.updateInterval - timeSinceLastUpdate);
        }
    }

    updateVisualization() {
        const labels = this.values.map(point => point.x);
        const data = this.values.map(point => point.y);

        this.chart.data.labels = labels;
        this.chart.data.datasets[0].data = data;
        this.chart.update();

        this.lastUpdateTime = Date.now();
        this.timeoutId = null;
    }
}

class FPSCounter {
    constructor(name, periodicLog=2000) {
        this.frames = 0;
        this.lastFPS = -1;
        this.startTime = performance.now();

        this.periodicLog = periodicLog;
        this.lastLog = performance.now();

        this.name = name;
        this.vislog = new VisLogger({
            name: name,
            tab: "Debug",
            xLabel: `Time since start (sec)`,
            yLabel: "FPS",
        });
    }

    update() {
        this.frames++;
        const currentTime = performance.now();
        const elapsedTime = currentTime - this.lastLog;
        
        if (currentTime - this.lastLog > this.periodicLog) {
            this.lastFPS = this.frames / (elapsedTime / 1000);
            this.lastLog = currentTime;
            this.log();
            this.frames = 0;
        }
        
    }

    log() {
        // console.log(`${this.name} - moving avg FPS: ${this.lastFPS.toFixed(2)}`);
        // console.log(`Number of tensors: ${tf.memory().numTensors}`);
        let elapsedTime = Math.floor((performance.now() - this.startTime)/1000);
        this.vislog.push({x: elapsedTime, y: this.lastFPS});
    }
}