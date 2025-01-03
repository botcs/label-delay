////////////////////////////////////////
// CONSTANTS
////////////////////////////////////////
const IMAGE_SIZE = 32;
const IMAGE_CHANNELS = 3;
const NUM_CLASSES = 3;
const NUM_FEATURES = 3*3;
const UNLABELED_IDX = 42;

const MEMORY_SIZE = 10;
const PENDING_SIZE = 1;

const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
let REFRESH_RATE = 30;

////////////////////////////////////////
// Machine Learning params
////////////////////////////////////////
const LR = 0.01;
// const MOMENTUM = 0.3;
const OPTIMIZER = tf.train.sgd(LR);
// const OPTIMIZER = tf.train.momentum(LR, MOMENTUM);
// const OPTIMIZER = tf.train.adam(LR);
// const ARCHITECTURE = "mobilenetv3";
// const ARCHITECTURE = "cnn_base";
// const ARCHITECTURE = "resnet18";
const ARCHITECTURE = "linear";
const TRAIN_REPEAT_INTERVAL = 1000;


////////////////////////////////////////
// BACKEND
////////////////////////////////////////
class DataEntry {
    static count = 0;
    constructor(inData, outData = null, label = UNLABELED_IDX) {tf.tidy(() => {
        this.inData = inData; // An image or a video frame

        this.inData.Tensor = tf.variable(inData.Tensor);

        // A [NUM_CLASSES] tensor representing the class logits
        this.logit = tf.variable(tf.squeeze(outData[0]));

        // A [NUM_CLASES] tensor representing the class probabilities
        this.pred = tf.variable(tf.softmax(this.logit));

        // A [NUM_FEATURES] tensor representing the feature vector
        const feat = outData[1];
        this.feat = tf.variable(tf.squeeze(feat.div(feat.norm(2))));

        // cast to integer
        this.label = parseInt(label);

        this.id = DataEntry.count;


        this.dataCard = null;
        DataEntry.count++;
    })}


    dispose() {
        this.inData.Tensor.dispose();
        this.logit.dispose();
        this.pred.dispose();
        this.feat.dispose();
    }

    clone() {
        return tf.tidy(() => new DataEntry(
            {dataURL: this.inData.dataURL, Tensor: this.inData.Tensor.clone()},
            [this.pred.clone(), this.feat.clone()],
            this.label
        ));
    }

    setLabel(label) {
        this.label = parseInt(label);
    }
}

class DataHandler {
    static instance = null;

    constructor({memorySize = MEMORY_SIZE, pendingSize = PENDING_SIZE} = {}) {
        if (DataHandler.instance !== null) {
            throw new Error("DataHandler instance already exists");
        }
        DataHandler.instance = this;

        this.memorySize = memorySize;
        this.pendingSize = pendingSize;

        this.currentEntry = null;

        this.memoryEntries = [];
        this.pendingEntries = [];

        // Store the cosine similarities
        this.similarities = tf.variable(tf.zeros([this.pendingSize, this.memorySize]));

        // Store softmax scores
        this.scores = tf.variable(tf.zeros([this.pendingSize, this.memorySize]));

        // Callback for new memory entry
        this.onNewMemoryEntry = null;

        // Callback for removing memory entry
        this.onMemoryFull = null;

        // This is storing all the provided raw images
        // with their corresponding labels for potentially saving it later
        this.labeledImages = [];
    }

    updateMemoryAndPendingSize({
        memorySize = this.memorySize,
        pendingSize = this.pendingSize
    }) {
        if (memorySize === this.memorySize && pendingSize === this.pendingSize) {
            return;
        }
        console.log('updating memory and pending size:', memorySize, pendingSize);

        this.memorySize = memorySize;
        this.pendingSize = pendingSize;

        // If the memory size is reduced, remove the oldest entries
        while (this.memoryEntries.length > memorySize) {
            const oldestEntry = this.memoryEntries.pop();
            oldestEntry.dispose();
        }

        // If the pending size is reduced, remove the oldest entries
        while (this.pendingEntries.length > pendingSize) {
            console.log('removing pending entry, size:', this.pendingEntries.length);
            const oldestEntry = this.pendingEntries.pop();
            oldestEntry.dispose();
        }

        // Reinitialize the similarities and scores variables
        this.similarities.dispose();
        this.similarities = tf.variable(tf.zeros([pendingSize, memorySize]));

        this.scores.dispose();
        this.scores = tf.variable(tf.zeros([pendingSize, memorySize]));

    }


    updateCurrentEntry(newEntry) {
        if (this.currentEntry !== null) {
            this.currentEntry.dispose();
        }
        this.currentEntry = newEntry;
    }


    unshiftAndPop(matrix, vector, dim) {
        // Add the vector to the matrix and remove the last vector
        // along the specified dimension

        // const requiredShape = matrix.shape.slice(0, dim)
            // .concat(matrix.shape.slice(dim+1));

        const requiredShape = Array.from(matrix.shape);
        requiredShape[dim] = 1;

        vector = vector.expandDims(dim);

        if (!vector.shape.every((d, i) => d === requiredShape[i])) {
            console.log(vector.shape);
            console.log(matrix.shape);
            throw new Error('The dimensions of the vector and matrix do not match');
        }

        const sliceStart = Array(matrix.shape.length).fill(0);
        // const sliceEnd = matrixTensor.shape.clone();
        const sliceEnd = Array.from(matrix.shape);
        sliceEnd[dim] -= 1;

        const popMatrix = matrix.slice(sliceStart, sliceEnd);
        const newMatrix = tf.concat([vector, popMatrix], dim);

        return newMatrix;
    }

    // In this app we only update either rows or lines never the entire matrix
    computeSingleSimilarity(queryEntry, keyEntries, expectedLength) {
        if (keyEntries.length > expectedLength) {
            throw new Error('The length of the key entries is larger than expected');
        }

        if (keyEntries.length === 0) {
            return tf.zeros([expectedLength]);
        }

        const queryFeat = queryEntry.feat;
        let keyFeats = [];
        for (let i = 0; i < keyEntries.length; i++) {
            keyFeats.push(keyEntries[i].feat);
        }

        keyFeats = tf.stack(keyFeats);
        let sim = tf.dot(keyFeats, queryFeat);

        // If the length of the key entries is smaller than expected
        // we need to pad the similarity matrix with zeros
        if (keyEntries.length < expectedLength) {
            const padLength = expectedLength - keyEntries.length;
            const pad = tf.zeros([padLength]);
            sim = tf.concat([sim, pad]);
        }

        return sim
    }


    addDataEntry(dataEntry) { tf.tidy(() => {
        this.pendingEntries.unshift(dataEntry);

        // Add new row to the top and remove bottom row
        const simPenToMem = this.computeSingleSimilarity(
                dataEntry, this.memoryEntries, this.memorySize
            )
        this.similarities.assign(this.unshiftAndPop(
            this.similarities, simPenToMem, 0
        ));

        // If the capacity of the pending entries is exceeded
        if (this.pendingEntries.length > this.pendingSize) {
            // Remove the oldest pending entry
            const transitionEntry = this.pendingEntries.pop();

            // If labeled, add to the memory entries
            if (transitionEntry.label !== UNLABELED_IDX) {
                this.memoryEntries.unshift(transitionEntry);

                // Add new column to the left and remove right column
                const simMemToPen = this.computeSingleSimilarity(
                    transitionEntry, this.pendingEntries, this.pendingSize
                );
                this.similarities.assign(this.unshiftAndPop(
                    this.similarities, simMemToPen, 1
                ));

                // Call the New Memory Entry callback
                if (this.onNewMemoryEntry !== null) {
                    this.onNewMemoryEntry();
                }

                this.labeledImages.push({
                    dataURL: transitionEntry.inData.dataURL,
                    label: transitionEntry.label
                });

            } else {
                transitionEntry.dispose();
            }
        }

        // If the capacity of the memory entries is exceeded
        if (this.memoryEntries.length > this.memorySize) {
            // Remove the oldest memory entry
            const oldestEntry = this.memoryEntries.pop();
            oldestEntry.dispose();

            if (this.onMemoryFull !== null) {
                this.onMemoryFull();
            }
        }

        // Update the softmax scores
        this.scores.assign(tf.softmax(this.similarities, 1));
    })};

    async updateSimilaritiesRow(rowIndex) {
        // Lazy update of the similarities
        const simPenToMem = this.computeSingleSimilarity(
            this.pendingEntries[rowIndex],
            this.memoryEntries,
            this.memorySize
        );
        // use tf assign to update the similarities
        const beforeRow = this.similarities.slice([0, 0], [rowIndex, -1]); // Slice before the row
        const afterRow = this.similarities.slice([rowIndex + 1, 0], [-1, -1]); // Slice after the row

        // Concatenate the parts with the new row
        const similarities = tf.concat([beforeRow, tf.reshape(simPenToMem, [1, -1]), afterRow], 0);
        this.similarities.assign(similarities);

        // Update the softmax scores
        this.scores.assign(tf.softmax(this.similarities, 1));
    }

    updateSimilarities() {
        // Full update of the similarities
        const keyEntries = this.memoryEntries;
        const queryEntries = this.pendingEntries;
        const keyFeats = [];
        const queryFeats = [];
        for (let i = 0; i < keyEntries.length; i++) {
            keyFeats.push(keyEntries[i].feat);
        }
        for (let i = 0; i < queryEntries.length; i++) {
            queryFeats.push(queryEntries[i].feat);
        }
        const keyFeatsTensor = tf.stack(keyFeats);
        const queryFeatsTensor = tf.stack(queryFeats);
        const similarities = tf.dot(queryFeatsTensor, keyFeatsTensor.transpose());

        // pad the similarities with zeros
        const padLength = Math.max(0, this.memorySize - keyEntries.length);

        const pad = tf.zeros([queryEntries.length, padLength]);
        this.similarities.assign(tf.concat([similarities, pad], 1));

        this.scores.assign(tf.softmax(this.similarities, 1));
    }

    recomputeFeatures(model) {
        // Recompute the features of the pending entries
        for (let i = 0; i < this.pendingEntries.length; i++) {
            const dataEntry = this.pendingEntries[i];
            const inData = dataEntry.inData;
            const outData = model.predict(inData.Tensor);
            dataEntry.pred.assign(tf.squeeze(outData[0]));
            const feat = outData[1];
            dataEntry.feat.assign(tf.squeeze(feat.div(feat.norm(2))));
        }
        // Recompute the features of the memory entries
        for (let i = 0; i < this.memoryEntries.length; i++) {
            const dataEntry = this.memoryEntries[i];
            const inData = dataEntry.inData;
            const outData = model.predict(inData.Tensor);
            dataEntry.pred.assign(tf.squeeze(outData[0]));
            const feat = outData[1];
            dataEntry.feat.assign(tf.squeeze(feat.div(feat.norm(2))));
        }

        this.updateSimilarities();
    }
}

class ModelHandler{
    static instance = null;

    constructor() {
        if (ModelHandler.instance !== null) {
            throw new Error("ModelHandler instance already exists");
        }
        ModelHandler.instance = this;
        this.model = null;
        this.seed = 42;
        this.randomIdx = Math.floor(Math.random() * dataHandler.memoryEntries.length);
        this.randomIdx2 = Math.floor(Math.random() * dataHandler.memoryEntries.length);

        this.X1Policy = "random";
        this.X2Policy = "random2";
        this.updateFeatures = true;

        this.numTrainIterations = 0;
        this.numValIterations = 0;
        this.countCorrect = 0;
        this.valAccuracies = new VisLogger({
            name: "Validation Accuracy",
            tab: "Online Evaluation",
            yLabel: "Accuracy",
        });
        this.valOnlineAccuracies = new VisLogger({
            name: "Validation Online Accuracy",
            tab: "Online Evaluation",
            yLabel: "Online Accuracy",
        });
        this.valLosses = new VisLogger({
            name: "Validation Loss",
            tab: "Online Evaluation",
            yLabel: "Loss",
        });
        this.trainAccuracies = new VisLogger({
            name: "Training Accuracy",
            tab: "Training",
            yLabel: "Accuracy",
        });
        this.trainLosses = new VisLogger({
            name: "Training Loss",
            tab: "Training",
            yLabel: "Loss",
        });
    }

    async initializeModel({
        architecture = ARCHITECTURE,
        image_size = IMAGE_SIZE,
        image_channels = IMAGE_CHANNELS,
        num_classes = NUM_CLASSES,
        num_features = NUM_FEATURES,
        optimizer = OPTIMIZER
    } = {}) {
        Math.seedrandom('constant');
        console.log(`Seed: ${this.seed}`);
        console.log(`Initialising model with following parameters:`)
        console.log(`Architecture: ${architecture}`);
        console.log(`Input Image Size: ${image_size}`);
        console.log(`Image Channels: ${image_channels}`);
        console.log(`Number of Classes: ${num_classes}`);
        console.log(`Number of Features: ${num_features}`);
        console.log(`Optimizer: ${optimizer}`);

        this.model = new GenericModelConnector({
            architecture: architecture,
            image_size: image_size,
            image_channels: image_channels,
            num_classes: num_classes,
            num_features: num_features,
            optimizer: optimizer
        });
        await this.model.loadModel();

        this.numTrainIterations = 0;

        // Warm up model by optimizing one iteration on
        // a random data entry
        const warmupX = tf.randomNormal([3, image_size, image_size, image_channels]);
        const warmupY = tf.oneHot(tf.tensor1d([0, 1, 2], 'int32'), num_classes);
        const warmupData = {input: warmupX, labels: warmupY};
        this.updateModelParameters(warmupData, true);
        // Dispose
        warmupX.dispose()
        warmupY.dispose()


        similarityGridHandler.THETA_T.text(this.numTrainIterations)
            .append("tspan")
            .attr("dy", "0.5em")
            .text("=train(");

    }

    updateRandomSeed(seed) {
        this.seed = seed;
        Math.seedrandom(seed);
        console.log(`Random seed updated to: ${this.seed}`);
    }

    evaluateModel(memoryEntryIdx=0) {
        const memoryEntry = dataHandler.memoryEntries[memoryEntryIdx];
        const logit = memoryEntry.logit;
        const label = memoryEntry.label;
        const pred = memoryEntry.pred;

        const lossFunction = tf.losses.softmaxCrossEntropy;
        const loss = lossFunction(
            tf.oneHot([label], NUM_CLASSES),
            tf.reshape(logit, [1, NUM_CLASSES])
        ).dataSync()[0];
        this.valLosses.push(loss);

        const predIdx = pred.argMax().dataSync()[0];
        const labelIdx = memoryEntry.label;

        const accuracy = predIdx === labelIdx ? 1. : 0.;
        this.valAccuracies.push(accuracy);

        this.countCorrect += accuracy;
        this.numValIterations++;
        const onlineAccuracy = this.countCorrect / this.numValIterations;
        this.valOnlineAccuracies.push(onlineAccuracy);
    }

    changeOptimizer(optimizer) {
        this.model.optimizer = optimizer;
    }

    updateRandomIdxs() {
        this.randomIdx = Math.floor(Math.random() * dataHandler.memoryEntries.length);
        this.randomIdx2 = Math.floor(Math.random() * dataHandler.memoryEntries.length);
    }


    async trainModel() {tf.tidy(() => {
        // Assert that at least one memory entry is present
        if (dataHandler.memoryEntries.length === 0) {
            throw new Error("At least one labeled data entry is required");
        }

        this.randomIdx = Math.floor(Math.random() * dataHandler.memoryEntries.length);
        this.randomIdx2 = Math.floor(Math.random() * dataHandler.memoryEntries.length);


        // Read the IWM index from DataHandler
        const iwmIdx = tf.tidy(() => dataHandler.scores.argMax(1).dataSync()[0]);


        const idxCatalog = {
            "newest": 0,
            "second-newest": d3.min([1, dataHandler.memorySize-1]),
            "iwm": iwmIdx,
            "random": this.randomIdx,
            "random2": this.randomIdx2
        }

        const idx1 = idxCatalog[this.X1Policy];
        const idx2 = idxCatalog[this.X2Policy];

        const indices = [idx1, idx2];


        const data = tf.tidy(() => {
            const input = [];
            const labels = [];
            for (let i = 0; i < indices.length; i++) {
                const dataEntry = dataHandler.memoryEntries[indices[i]];
                input.push(dataEntry.inData.Tensor);
                labels.push(parseInt(dataEntry.label));
            }
            return {
                input: tf.concat(input, 0),
                labels: tf.oneHot(labels, NUM_CLASSES)
            }
        });

        this.updateModelParameters(data);

        data.input.dispose();
        data.labels.dispose();

        this.numTrainIterations++;
        similarityGridHandler.THETA_T.text(this.numTrainIterations)
            .append("tspan")
            .attr("dy", "0.5em")
            .text("=train(");


        if (this.updateFeatures) {
            // Update all the features
            tf.tidy(() => {
                dataHandler.recomputeFeatures(this.model);
            });
        }
    })};

    updateModelParameters(data, warmup=false){
        let optimizer;
        if (warmup){
           optimizer = tf.train.adam(0.0);
        } else {
            optimizer = this.model.optimizer;
        }

        let accuracy = tf.tidy(() => {
            const features = this.model.forwardBackbone(data.input);
            const logits = this.model.forwardHead(features)[0];
            const pred = tf.softmax(logits);
            const correct = tf.equal(tf.argMax(pred, 1), tf.argMax(data.labels, 1));
            return correct.sum().dataSync()[0] / data.labels.shape[0];
        });
        this.trainAccuracies.push(accuracy);

        const lossFunction = tf.losses.softmaxCrossEntropy;
        const useFrozenBackbone = this.model.architecture.includes("mobilenet");
        if (useFrozenBackbone) {
            const features = this.model.forwardBackbone(data.input);
            // Just exclude the backbone from the optimization
            optimizer.minimize(() => {
                const logits = this.model.forwardHead(features)[0];
                const loss = lossFunction(data.labels, logits);
                if (warmup){
                    console.log(`Warmup complete!`);
                } else {
                    console.log(`Training iteration: ${this.numTrainIterations} - Loss: ${loss.dataSync()}`);
                }
                this.trainLosses.push(loss.dataSync()[0]);
                return loss;
            });
        } else {
            optimizer.minimize(() => {
                const features = this.model.forwardBackbone(data.input);
                const logits = this.model.forwardHead(features)[0];
                const loss = lossFunction(data.labels, logits);
                if (warmup){
                    console.log(`Warmup complete!`);
                } else {
                    console.log(`Training iteration: ${this.numTrainIterations} - Loss: ${loss.dataSync()}`);
                }
                this.trainLosses.push(loss.dataSync()[0]);
                return loss;
            });
        }
    }
}

////////////////////////////////////////
// FRONTEND
////////////////////////////////////////
class DataCard {
    // A datacard is a visual representation of a data entry
    static unitSize = 100;
    static animationDuration = 100;

    static layouts = {
        // [I | F]
        horizontal: {
            image: {x: 0, y: 0},
            feature: {x: DataCard.unitSize, y: 0},
            shape: {width: 2*DataCard.unitSize, height: DataCard.unitSize}
        },
        // [  | F]
        // [I |  ]
        diagonal: {
            image: {x: 0, y: DataCard.unitSize},
            feature: {x: DataCard.unitSize, y: 0},
            shape: {width: 2*DataCard.unitSize, height: 2*DataCard.unitSize}
        },
        // [F]
        // [I]
        vertical: {
            image: {x: 0, y: DataCard.unitSize},
            feature: {x: 0, y: 0},
            shape: {width: DataCard.unitSize, height: 2*DataCard.unitSize}
        }
    }

    static maxCircleRadius = DataCard.unitSize / Math.sqrt(NUM_FEATURES) / 2 * .8;

    constructor(dataEntry, position = {x: 0, y: 0}, orientation="horizontal") {
        dataEntry.dataCard = this;

        this.dataEntry = dataEntry;
        this.position = position;
        this.orientation = orientation;
        this.layout = DataCard.layouts[orientation];
        this.renderPromise = null;
    }

    async createDOM(parentGroup) {
        if (this.renderPromise !== null) {
            return this.renderPromise;
        }
        this.renderPromise = this._createDOM(parentGroup);
        await this.renderPromise;
        this.renderPromise = null;
    }

    async _createDOM(parentGroup) {
        const mainGroup = parentGroup
            .append("g");

        const clip = mainGroup.append("defs")
            .append("clipPath")
            .attr("id", "clip-rounded-rect")
            .append("rect");

        mainGroup.classed("datacard", true)
            .classed("category-" + this.dataEntry.label, true)
            .attr("data-id", this.dataEntry.id)
            .attr("transform", `translate(${this.position.x}, ${this.position.y})`);


        // The card has a background rounded 2x1 rectangle
        // The image on the top row is clipped to the rounded rectangle
        // The features in the bottom row are shown as a 3x3 grid of circles
        // The label is represented by the color of the colour theme of the card

        const background = mainGroup.append("rect")
            .classed("datacard-background", true)
            .attr("width", this.layout.shape.width)
            .attr("height", this.layout.shape.height)
            .attr("rx", 10)
            .attr("ry", 10);

        const imgX = DataCard.unitSize*.05;
        const imgY = DataCard.unitSize*.05;
        const imgWidth = DataCard.unitSize*.9;
        const imgHeight = DataCard.unitSize*.9;

        clip.attr("x", imgX)
            .attr("y", imgY)
            .attr("width", imgWidth)
            .attr("height", imgHeight)
            .attr("rx", 10)
            .attr("ry", 10);

        const imageGroup = mainGroup.append("g")
            .attr("transform", `translate(${this.layout.image.x}, ${this.layout.image.y})`);

        imageGroup.append("image")
            .classed("datacard-image", true)
            .attr("xlink:href", this.dataEntry.inData.dataURL)
            .attr("x", imgX)
            .attr("y", imgY)
            .attr("width", imgWidth)
            .attr("height", imgHeight)
            .attr("clip-path", "url(#clip-rounded-rect)");

        const featureGroup = mainGroup.append("g")
            .classed("feature-group", true)
            .attr("transform", `translate(${this.layout.feature.x}, ${this.layout.feature.y})`);


        let feat = await this.dataEntry.feat.data();

        // since feat is in [-inf, inf] we need to map it to [0, 1]
        const maxFeatVal = d3.max(feat);
        const minFeatVal = d3.min(feat);
        feat = feat.map(d => (d - minFeatVal) / (maxFeatVal - minFeatVal));

        const featPerRow = Math.ceil(Math.sqrt(feat.length));
        const pos = d3.scalePoint()
            .domain(d3.range(featPerRow))
            .range([0, DataCard.unitSize])
            .padding(.5);

        featureGroup.selectAll("circle")
            .data(feat)
            .join("circle")
            .attr("cx", (d, i) => pos(i % featPerRow))
            .attr("cy", (d, i) => pos(Math.floor(i / featPerRow)))
            .attr("r", (d, i) => d * DataCard.maxCircleRadius)

        this.mainGroup = mainGroup;
        this.background = background;
        this.imageGroup = imageGroup;
        this.featureGroup = featureGroup;
        this.rendered = true;
    }

    async updateDOM() {
        if (!this.rendered) {
            return;
        }
        if (this.renderPromise !== null) {
            return this.renderPromise;
        }
        this.renderPromise = this._updateDOM();
        await this.renderPromise;
        this.renderPromise = null;
    }
    async _updateDOM() {
        // update position
        this.mainGroup.attr(
            "transform", `translate(${this.position.x}, ${this.position.y})`
        );

        // Update label
        // remove previous label
        this.mainGroup.attr("class", "datacard category-" + this.dataEntry.label);

        // Update the image and the feature vector
        this.imageGroup.select("image")
            .attr("xlink:href", this.dataEntry.inData.dataURL);

        let feat = await this.dataEntry.feat.data();
        const maxFeatVal = d3.max(feat);
        const minFeatVal = d3.min(feat);
        feat = feat.map(d => (d - minFeatVal) / (maxFeatVal - minFeatVal));

        this.featureGroup.selectAll("circle")
            .data(feat)
            .join("circle")
            .attr("r", (d, i) => d * DataCard.maxCircleRadius)
    }


    async changeOrientation(orientation) {
        if (orientation === this.orientation) {
            return;
        }
        if (orientation === "diagonal") {
            // Raise error because this is a temporary state
            throw new Error("diagonal orientation is not allowed to be set");
        }

        // First change to diagonal
        const diagonalLayout = DataCard.layouts["diagonal"];
        this.background.transition()
            .duration(DataCard.animationDuration)
            .attr("width", diagonalLayout.shape.width)
            .attr("height", diagonalLayout.shape.height);

        this.imageGroup.transition()
            .duration(DataCard.animationDuration/2)
            .attr("transform", `translate(${diagonalLayout.image.x}, ${diagonalLayout.image.y})`);


        await this.featureGroup.transition()
            .duration(DataCard.animationDuration)
            .attr("transform", `translate(${diagonalLayout.feature.x}, ${diagonalLayout.feature.y})`)
            .end();

        this.layout = DataCard.layouts[orientation];
        this.orientation = orientation;

        // transition to diagonal
        this.background.transition()
            .duration(DataCard.animationDuration)
            .attr("width", this.layout.shape.width)
            .attr("height", this.layout.shape.height);

        this.imageGroup.transition()
            .duration(DataCard.animationDuration)
            .attr("transform", `translate(${this.layout.image.x}, ${this.layout.image.y})`);

        this.featureGroup.transition()
            .duration(DataCard.animationDuration)
            .attr("transform", `translate(${this.layout.feature.x}, ${this.layout.feature.y})`);

    }


    async changePosition(centerPosition) {
        // Move the card such that the featureGroup is centered at the position
        this.position = {
            x: centerPosition.x - this.layout.feature.x - DataCard.unitSize/2,
            y: centerPosition.y - this.layout.feature.y - DataCard.unitSize/2
        }
        const x = this.position.x;
        const y = this.position.y;
        await this.mainGroup.transition()
            .duration(DataCard.animationDuration)
            .attr(
                "transform", `translate(${x}, ${y})`
            )
            .end();
    }

    removeDOM() {
        // Remove the DOM elements
        // use fading out animation
        this.mainGroup.transition()
            .duration(DataCard.animationDuration)
            .style("opacity", 0)
            .remove();
    }
}

class PredCard {
    constructor(dataEntry, position = {x: 0, y: 0}) {
        this.dataEntry = dataEntry;
        this.position = position;
        this.renderPromise = null;
    }

    async createDOM(parentGroup) {
        if (this.renderPromise !== null) {
            return this.renderPromise;
        }
        this.renderPromise = this._createDOM(parentGroup);
        await this.renderPromise;
        this.renderPromise = null;
    }

    async _createDOM(parentGroup) {
        const mainGroup = parentGroup
            .append("g")
            .attr("transform", `translate(${this.position.x}, ${this.position.y})`);

        const background = mainGroup.append("rect")
            .classed("predcard-background", true)
            .attr("width", DataCard.unitSize / NUM_CLASSES)
            .attr("height", DataCard.unitSize)
            .attr("rx", 10)
            .attr("ry", 10);

        this.mainGroup = mainGroup;
        this.background = background;
        this.rendered = true;
    }

    async updateDOM() {
        if (!this.rendered) {
            return;
        }
        if (this.renderPromise !== null) {
            return this.renderPromise;
        }
        this.renderPromise = this._updateDOM();
        await this.renderPromise;
        this.renderPromise = null;
    }
    async _updateDOM() {
        // represent each class with a circle of different color
        const pred = await this.dataEntry.pred.data();
        const maxPredVal = d3.max(pred);
        const circleRadii = [];
        for (let i = 0; i < pred.length; i++) {
            circleRadii.push(pred[i] / maxPredVal * DataCard.maxCircleRadius);
        }

        const fillColors = ["#f9cb9c", "#a4c2f4", "#b6d7a8"];
        const strokeColors = ["#e69138", "#3c78d8","#6aa84f"];

        // The circles are in a single column
        this.mainGroup.selectAll("circle")
            .data(circleRadii)
            .join("circle")
            .attr("cx", DataCard.unitSize / NUM_CLASSES / 2)
            .attr("cy", (d, i) => i * DataCard.unitSize / NUM_CLASSES + DataCard.unitSize / NUM_CLASSES / 2)
            .attr("r", d => d)
            .attr("fill", (d, i) => fillColors[i])
            .attr("stroke", (d, i) => strokeColors[i])
            .attr("stroke-width", 2);
    }
}

class PredCardHandler {
    static instance = null;

    constructor() {
        if (PredCardHandler.instance !== null) {
            throw new Error("PredCardHandler instance already exists");
        }
        PredCardHandler.instance = this;

        this.predCard = null;
        this.dataCard = null;
        this.renderPromise = null;

        this.isInitialized = false;

        this.predCardFPS = new FPSCounter("PredCard FPS");
    }

    async createDOM() {
        const offset = {x: 5, y: 5};
        const predSVG = d3.select("#prediction-card");

        this.mainGroup = predSVG.append("g");

        const currentDataEntry = dataHandler.currentEntry;
        this.dataCard = new DataCard(
            currentDataEntry,
            {x: offset.x, y: offset.y},
            "horizontal"
        );
        await this.dataCard.createDOM(this.mainGroup);

        this.predCard = new PredCard(
            currentDataEntry,
            {
                x: this.dataCard.layout.shape.width*1.05 + offset.x,
                y: offset.y,
            }
        );
        await this.predCard.createDOM(this.mainGroup);
        await this.predCard.updateDOM();

        this.mainGroup.append("text")
            .text("input")
            .attr("x", offset.x + DataCard.unitSize/2)
            .attr("y", offset.y + DataCard.unitSize*1.1)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "hanging")
            .attr("font-size", "1.5em");

        this.mainGroup.append("text")
            .text("feat")
            .attr("x", offset.x + DataCard.unitSize*1.5)
            .attr("y", offset.y + DataCard.unitSize*1.1)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "hanging")
            .attr("font-size", "1.5em");

        this.mainGroup.append("text")
            .text("pred")
            .attr("x", this.predCard.position.x + DataCard.unitSize/NUM_CLASSES/2)
            .attr("y", offset.y + DataCard.unitSize*1.1)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "hanging")
            .attr("font-size", "1.5em");

        this.isInitialized = true;
    }

    async updateDOM() {
        this.predCardFPS.update();
        if (!this.isInitialized) {
            await this.createDOM();
        }
        const currentDataEntry = dataHandler.currentEntry;
        this.dataCard.dataEntry = currentDataEntry;
        this.predCard.dataEntry = currentDataEntry;
        await this.dataCard.updateDOM();
        await this.predCard.updateDOM();
    }
}
// const predCardHandler = new PredCardHandler(dataHandler);


class SimilarityGridHandler {
    static instance = null;

    constructor({
        memorySize = MEMORY_SIZE,
        pendingSize = PENDING_SIZE,
        offset = {x: 0, y: 0}
    } = {}) {
        if (SimilarityGridHandler.instance !== null) {
            throw new Error("SimilarityGridHandler instance already exists");
        }

        if (DataHandler.instance === null) {
            throw new Error("DataHandler instance is required for SimilarityGridHandler");
        }

        if (ModelHandler.instance === null) {
            throw new Error("ModelHandler instance is required for SimilarityGridHandler");
        }

        SimilarityGridHandler.instance = this;

        // the memory entries are the labeled datacards
        // the pending entries are the unlabeled datacards
        // the width and height are used for the inner dimensions of table
        this.memorySize = memorySize;
        this.pendingSize = pendingSize;
        this.offset = offset;

        this.X1Policy = "random";
        this.X2Policy = "random2";

        this.memoryCards = [];
        this.pendingCards = [];
        this.mainDOMgroup = null;
    }

    async initialize() {
        if (dataHandler.memorySize !== this.memorySize) {
            throw new Error("Memory size mismatch between DataHandler and SimilarityGridHandler");
        }
        if (dataHandler.pendingSize !== this.pendingSize) {
            throw new Error("Pending size mismatch between DataHandler and SimilarityGridHandler");
        }

        this.svg = d3.select("#similarity-grid");

        // width is 1x[horizontal card] + (MEM+1)x[vertical card]
        this.boardWidth = DataCard.layouts["horizontal"].shape.width;
        this.boardWidth += (this.memorySize + 1) * DataCard.layouts["vertical"].shape.width;

        // height is PENDx[horizontal card] + 1x[vertical card]
        this.boardHeight = this.pendingSize * DataCard.layouts["horizontal"].shape.height;
        this.boardHeight += DataCard.layouts["vertical"].shape.height;

        this.equationHeight = DataCard.unitSize*2;

        // set the viewbox of the SVG
        this.svg.attr("viewBox", `0 0 ${this.boardWidth} ${this.boardHeight + this.equationHeight}`);
        this.renderPromise = null;

        this.setDOMPositions();
        if (this.mainDOMGroup == null) {
            this.mainDOMGroup = this.svg.append("g")
                .attr("id", "mainDOMGroup")
                .attr("transform", `translate(${this.offset.x}, ${this.offset.y})`);
        }

        // draw the grid
        this.grid = this.mainDOMGroup.append("g")
            .attr("id", "grid");
        this.grid.append("g")
            .attr("id", "horGrid")
            .selectAll("line")
            .data(this.gridX.domain())
            .join("line")
            .attr("x1", d => this.gridLineX(d))
            .attr("x2", d => this.gridLineX(d))
            .attr("y1", this.gridLineY(0))
            .attr("y2", this.gridLineY(this.pendingSize))
            .attr("stroke", "lightgrey")
            .attr("stroke-width", 1)
            .attr("stroke-dasharray", "5,5");

        this.grid.append("g")
            .attr("id", "verGrid")
            .selectAll("line")
            .data(this.gridY.domain())
            .join("line")
            .attr("y1", d => this.gridLineY(d))
            .attr("y2", d => this.gridLineY(d))
            .attr("x1", this.gridLineX(0))
            .attr("x2", this.gridLineX(this.memorySize))
            .attr("stroke", "lightgrey")
            .attr("stroke-width", 1)
            .attr("stroke-dasharray", "5,5");


        this.legendGroup = this.mainDOMGroup.append("g")
            .attr("id", "legendGroup");

        this.legendGroup.append("text")
            .text("Pending Entries")
            .attr("x", this.gridX(0) - DataCard.unitSize*.5)
            .attr("y", this.gridY(this.pendingSize) - DataCard.unitSize*.45)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "hanging")
            .attr("font-size", "1.5em")
            .attr("fill", "grey");

        this.legendGroup.append("text")
            .text("Memory Buffer")
            .attr("x", this.gridX(0) + DataCard.unitSize*.45)
            .attr("y", this.gridY(this.pendingSize) + DataCard.unitSize*.5)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "alphabetic")
            .attr("font-size", "1.5em")
            .attr("transform", `rotate(-90 ${this.gridX(0) + DataCard.unitSize*.45} ${this.gridY(this.pendingSize) + DataCard.unitSize*.5})`)
            .attr("fill", "grey");



        // Add group for the similarity matrix
        this.similarityGroup = this.mainDOMGroup.append("g")
            .attr("id", "similarityGroup");

        // Add group for the equation
        this.equationGroup = this.mainDOMGroup.append("g")
            .attr("id", "equationGroup");

        this.THETA_T = this.equationGroup.append("text")
            .attr("id", "THETA")
            .style("font-size", "3em")
            .style("text-anchor", "middle")
            .style("dominant-baseline", "hanging")
            .attr("x", this.boardWidth*0.3)
            .attr("y", this.boardHeight + DataCard.unitSize)
            .text("θ")
            .append("tspan")
            .attr("dy", "-0.5em")
            .text("0");
        this.THETA_T.append("tspan")
            .attr("dy", "0.5em")
            .text("=train(");

        this.X_1 = this.equationGroup.append("text")
            .attr("id", "X_RND")
            .style("font-size", "3em")
            .style("text-anchor", "middle")
            .style("dominant-baseline", "hanging")
            .attr("x", this.boardWidth*0.5)
            .attr("y", this.boardHeight + DataCard.unitSize)
            .text("X");
        this.X_1.append("tspan")
            .attr("dy", "0.5em")
            .text("1")
            .append("tspan")
            .attr("dy", "-0.5em")
            .text(",");
        this.X_2 = this.equationGroup.append("text")
            .attr("id", "X_IWM")
            .style("font-size", "3em")
            .style("text-anchor", "middle")
            .style("dominant-baseline", "hanging")
            .attr("x", this.boardWidth*0.7)
            .attr("y", this.boardHeight + DataCard.unitSize)
            .text("X");
        this.X_2.append("tspan")
            .attr("dy", "0.5em")
            .text("2")
            .append("tspan")
            .attr("dy", "-0.5em")
            .text(")");
    }

    setDOMPositions(padding = 0.0) {
        // we use a grid layout for the datacards
        // on the left we have the pending entries
        //      (top - newest -> bottom - oldest)
        // on the bottom we have the labeled entries
        //      (left - newest -> right - oldest)

        // in the middle we have the similarity matrix
        // between the pending entries and the labeled entries

        // The datacards are centered around the featureGroup
        // In horizontal the cards are [I | F]
        // therefore we need to offset the grid by 1.5 units on X
        // and 0.5 units on Y
        const startX = DataCard.unitSize*1.5;
        const startY = DataCard.unitSize*0.5;

        // The grid ends at the right and bottom border
        // therefore we need to subtract 0.5 units on X
        // and 1.5 units on Y because the cards are vertical
        const endX = this.boardWidth - DataCard.unitSize*0.5;
        const endY = this.boardHeight - DataCard.unitSize*1.5;


        // We add +1 to accommodate the cards
        const numRows = this.pendingSize + 1;
        const numCols = this.memorySize + 1;

        this.gridX = d3.scalePoint()
            .domain(d3.range(numCols))
            .range([startX, endX])
            .padding(padding);
        this.gridY = d3.scalePoint()
            .domain(d3.range(numRows))
            .range([startY, endY])
            .padding(padding);

        this.gridLineX = (i) => this.gridX(i) + DataCard.unitSize/2;
        this.gridLineY = (i) => this.gridY(i) - DataCard.unitSize/2;

        this.pendingDOMPositions = [];
        for (let i = 0; i < this.pendingSize; i++) {
            const position = { x: this.gridX(0), y: this.gridY(i) };
            this.pendingDOMPositions.push(position);
        }

        this.memoryDOMPositions = [];
        for (let i = 0; i < this.memorySize; i++) {
            const position = { x: this.gridX(i+1), y: this.gridY(this.pendingSize) };
            this.memoryDOMPositions.push(position);
        }

        this.transitionPosition = {
            x: this.gridX(0),
            y: this.gridY(this.pendingSize)
        };
    }

    createVertConnector(fromP, toP) {
        fromP.x = parseInt(fromP.x);
        fromP.y = parseInt(fromP.y);
        toP.x = parseInt(toP.x);
        toP.y = parseInt(toP.y);
        const path = d3.path();
        path.moveTo(fromP.x, fromP.y);
        const sharpness = 0.9;
        const control1X = fromP.x;
        const control1Y = fromP.y + sharpness * (toP.y - fromP.y);
        const control2X = toP.x;
        const control2Y = toP.y - sharpness * (toP.y - fromP.y);

        // path.quadraticCurveTo(controlX, controlY, toP.x, toP.y);
        path.bezierCurveTo(control1X, control1Y, control2X, control2Y, toP.x, toP.y);

        return path.toString();
    }

    async renderDataCards() {
        // Re-renders all dataCards in the DOM
        // This is used when the model is trained
        // and the features are re-computed

        const renderPromises = [];
        // Update the pending entries
        for (let i = 0; i < this.pendingCards.length; i++) {
            const p = this.pendingCards[i].updateDOM();
            renderPromises.push(p);
        }
        // Update the memory entries
        for (let i = 0; i < this.memoryCards.length; i++) {
            const p = this.memoryCards[i].updateDOM();
            renderPromises.push(p);
        }
        await Promise.all(renderPromises);
    }

    async renderSimilarities() {
        if (this.memoryCards.length === 0) {
            return;
        }
        let scores = await dataHandler.scores.array();

        // Trim cols to the number of memory entries
        scores = scores.map(row => row.slice(0, this.memoryCards.length));

        const numRows = scores.length;
        const numCols = scores[0].length;
        const maxScores = scores.map(row => d3.max(row)); // Maximum score per row

        const barPadding = 0.3;
        const barWidth = DataCard.unitSize * (1 - barPadding);
        const barHeight = DataCard.unitSize * 0.95;

        const barHeightScale = (d, i) => d3.scaleLinear()
            .domain([0, maxScores[Math.floor(i / numCols)]])
            .range([0, barHeight])(d);

        const barX = (j) => this.gridX(j % numCols + 1) - barWidth / 2;
        // Adjusted barY to start from the baseline
        const barY = (d, i) => {
            let y = this.gridY(Math.floor(i / numCols));
            y += DataCard.unitSize * 0.5;
            y -= barHeightScale(d, i);
            return y;
        };

        const maxIndices = scores.map(row => d3.maxIndex(row));

        function barOpacity(d, i) {
            if (i % numCols === maxIndices[Math.floor(i / numCols)]) {
                return 1.0;
            }
            return .69;
        }

        // find out if new values are added
        let duration = 10;
        if (this.similarityGroup.selectAll("rect").size() < scores.flat().length) {
            duration = 0;
        }

        function color(d, i) {
            if (i % numCols === maxIndices[Math.floor(i / numCols)]) {
                const categoryColors = ["#f9cb9c", "#a4c2f4", "#b6d7a8"];
                const label = dataHandler.memoryEntries[i % numCols].label;
                return categoryColors[label];
            }
            return "#c5c5c3";
        }

        this.similarityGroup.selectAll("rect")
            .data(scores.flat())
            .join("rect")
            .attr("width", barWidth)
            .transition()
            .duration(duration)
            .attr("x", (d, i) => barX(i))
            .attr("y", (d, i) => barY(d, i)) // Updated y attribute
            .attr("height", (d, i) => barHeightScale(d, i)) // Updated height attribute
            .attr("fill", (d, i) => color(d, i))
            .style("opacity", (d, i) => barOpacity(d, i))
            .attr("rx", 10)
            .attr("ry", 10);

        // Draw arrows from max similarity to the equation
        if (this.arrowX1 === undefined) {
            this.arrowX1 = this.svg.append("path")
                .attr("stroke", "red")
                .attr("stroke-width", 6)
                .attr("fill", "none");
        }
        if (this.arrowX2 === undefined) {
            this.arrowX2 = this.svg.append("path")
                .attr("stroke", "red")
                .attr("stroke-width", 6)
                .attr("fill", "none");
        }


        // Define memory indices to pick from depending on polciy
        const idxCatalog = {
            "newest": 0,
            "second-newest": d3.min([1, this.memorySize - 1]),
            "iwm": d3.min([maxIndices[0], this.memorySize - 1]),
            "random": modelHandler.randomIdx,
            "random2": modelHandler.randomIdx2
        }
        const X1Idx = idxCatalog[this.X1Policy];
        const X1Card = this.memoryCards[X1Idx];
        const startX1 = {
            x: X1Card.position.x + X1Card.layout.shape.width / 2,
            y: this.gridY(this.pendingSize) + DataCard.unitSize * 1.5
        }
        const endX1 = {
            x: parseInt(this.X_1.attr("x")),
            y: parseInt(this.X_1.attr("y"))
        }
        this.arrowX1.transition()
        .duration(100)
        .attr("d", this.createVertConnector(startX1, endX1));


        // start is the bottom of the selected MemoryCard
        // const X2Idx = d3.min([maxIndices[0], this.memorySize - 1]);
        const X2Idx = idxCatalog[this.X2Policy];
        const X2Card = this.memoryCards[X2Idx];
        if (X2Card !== undefined) {
            const start = {
                x: X2Card.position.x + X2Card.layout.shape.width / 2,
                y: this.gridY(this.pendingSize) + DataCard.unitSize * 1.5
            }
            // end is the top of the "X_IWM"
            const end = {
                x: parseInt(this.X_2.attr("x")),
                y: parseInt(this.X_2.attr("y"))
            }
            this.arrowX2.transition()
                .duration(100)
                .attr("d", this.createVertConnector(start, end));
        }

    }

    async updateMemorySize(newSize) {
        if (newSize === this.memorySize) {
            return;
        }
        this.memorySize = newSize;

        // remove the oldest memory entries
        while (this.memoryCards.length > this.memorySize) {
            const oldestMemoryCard = this.memoryCards.pop();
            await oldestMemoryCard.removeDOM();
        }

        this.grid.remove();
        this.equationGroup.remove();
        this.similarityGroup.remove();
        this.legendGroup.remove();
        this.initialize();

        // update the positions of the memory entries
        await this.updateMemoryPositions();

        await this.renderSimilarities();
    }

    async updatePendingSize(newSize) {
        if (newSize === this.pendingSize) {
            return;
        }
        this.pendingSize = newSize;

        // remove the oldest pending entries
        while (this.pendingCards.length > this.pendingSize) {
            const oldestPendingCard = this.pendingCards.pop();
            await oldestPendingCard.removeDOM();
        }

        this.grid.remove();
        this.equationGroup.remove();
        this.similarityGroup.remove();
        this.legendGroup.remove();
        this.initialize();

        // update the positions of the pending entries
        await this.updatePendingPositions();
        await this.updateMemoryPositions();
        await this.renderSimilarities();
    }

    async addDataCard(dataEntry) {
        if (this.renderPromise !== null) {
            return this.renderPromise;
        }
        this.renderPromise = this._addDataCard(dataEntry);
        await this.renderPromise;
        this.renderPromise = null;
    }

    async _addDataCard(dataEntry) {
        // Add card to the pending entries
        const dataCard = new DataCard(dataEntry);
        await dataCard.createDOM(this.mainDOMGroup);

        this.pendingCards.unshift(dataCard);

        let transitionCard = null
        if (this.pendingCards.length > this.pendingSize) {
            transitionCard = this.pendingCards.pop();
        }
        const asyncMoveCall = this.updatePendingPositions();


        if (transitionCard !== null) {
            // if labeled, add to the memory entries
            if (transitionCard.dataEntry.label !== UNLABELED_IDX) {
                this.memoryCards.unshift(transitionCard);
                await Promise.all([
                    transitionCard.changePosition(this.transitionPosition),
                    transitionCard.changeOrientation("vertical"),
                    asyncMoveCall
                ]);
            } else {
                // if unlabeled, remove from the DOM
                await Promise.all([
                    transitionCard.changePosition(this.transitionPosition),
                    asyncMoveCall
                ]);
                transitionCard.removeDOM();
            }
        }
        await asyncMoveCall;

        this.updateMemoryPositions();
        if (this.memoryCards.length > 0 && this.pendingCards.length == this.pendingSize) {
            this.renderSimilarities();
        }

        // If the capacity of the memory entries is exceeded
        // remove the oldest memory entry
        if (this.memoryCards.length > this.memorySize) {
            const oldestMemoryCard = this.memoryCards.pop();
            oldestMemoryCard.removeDOM();
        }
    }

    async updatePendingPositions() {
        const asyncCalls = [];
        // Update the positions of the pending entries
        for (let i = 0; i < this.pendingCards.length; i++) {
            const pendingCard = this.pendingCards[i];
            const position = this.pendingDOMPositions[i];
            const asyncCall = pendingCard.changePosition(position);
            asyncCalls.push(asyncCall);
        }
        await Promise.all(asyncCalls);
    }
    async updateMemoryPositions() {
        const asyncCalls = [];
        // Update the positions of the memory entries
        for (let i = 0; i < this.memoryCards.length; i++) {
            const memoryCard = this.memoryCards[i];
            const position = this.memoryDOMPositions[i];
            const asyncCall = memoryCard.changePosition(position);
            asyncCalls.push(asyncCall);
        }
        await Promise.all(asyncCalls);
    }

    getThumbnailClientSize() {
        if (this.pendingCards.length === 0) {
            return {width: IMAGE_SIZE, height: IMAGE_SIZE};
        }
        const rect = this.pendingCards[0]
            .imageGroup
            .node()
            .getBoundingClientRect();

        return {
            width: rect.width,
            height: rect.height
        };
    }
}
// const similarityGridHandler = new SimilarityGridHandler();


function frameToTensor(source) {
    // Read frame
    let frame = tf.browser.fromPixels(source)

    // Resize
    // frame = frame.resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE]);
    frame = tf.image.resizeBilinear(frame, [IMAGE_SIZE, IMAGE_SIZE]);

    // Add batch dimension
    frame = frame.expandDims(0);

    // Normalize to [-1, 1]
    frame = frame.div(tf.scalar(128)).sub(tf.scalar(1));

    // Normalize to [0, 1]
    // frame = frame.toFloat().div(tf.scalar(255));

    return frame;
}


function captureWebcam() {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('aux-canvas');
    const ctx = canvas.getContext('2d');
    const thumbnailSize = 224;

    // Set the canvas size to the output size
    canvas.width = thumbnailSize;
    canvas.height = thumbnailSize;

    // Calculate the coordinates to crop the video
    const videoAspectRatio = video.videoWidth / video.videoHeight;
    const outputAspectRatio = thumbnailSize / thumbnailSize;

    let sourceWidth, sourceHeight, sourceX, sourceY;

    if (videoAspectRatio > outputAspectRatio) {
        // Video is wider than the desired aspect ratio
        sourceHeight = video.videoHeight;
        sourceWidth = video.videoHeight * outputAspectRatio;
        sourceX = (video.videoWidth - sourceWidth) / 2;
        sourceY = 0;
    } else {
        // Video is taller than the desired aspect ratio
        sourceWidth = video.videoWidth;
        sourceHeight = video.videoWidth / outputAspectRatio;
        sourceX = 0;
        sourceY = (video.videoHeight - sourceHeight) / 2;
    }

    // Mirror the image horizontally
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);

    // Draw the cropped area onto the canvas
    ctx.drawImage(
        video,
        sourceX,
        sourceY,
        sourceWidth,
        sourceHeight,
        0,
        0,
        thumbnailSize,
        thumbnailSize
    );

    const dataURL = canvas.toDataURL('image/jpeg', 0.5);
    const dataTensor = tf.tidy(() => frameToTensor(canvas));
    // const dataTensor = tf.tidy(() => frameToTensor(video));

    const frame = {
        dataURL: dataURL,
        Tensor: dataTensor
    };
    return frame;
}

async function loadImage(url) {
    // Offline replacement of captureWebcam
    const img = await new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = url;
    });

    const dataURL = url;
    const dataTensor = tf.tidy(() => frameToTensor(img));

    const frame = {
        dataURL: dataURL,
        Tensor: dataTensor
    };
    return frame;
}

async function createDataCard(label = UNLABELED_IDX, url = null) {
    if (similarityGridHandler.renderPromise !== null) {
        await similarityGridHandler.renderPromise;
    }
    let inData;
    if (url !== null) {
        inData = await loadImage(url);
    }

    const dataEntry = tf.tidy(() => {
        if (url === null) {
            inData = captureWebcam();
        }
        const outData = modelHandler.model.predict(inData.Tensor);
        const dataEntry = new DataEntry(inData, outData, label);
        dataHandler.addDataEntry(dataEntry);
        return dataEntry;
    });

    await similarityGridHandler.addDataCard(dataEntry);
    console.log(`Added datacard with label ${label}`);
}


class EventHandler {
    static instance = null;
    constructor({memorySize = MEMORY_SIZE, pendingSize = PENDING_SIZE} = {}) {
        if (EventHandler.instance !== null) {
            throw new Error("EventHandler instance already exists");
        }
        EventHandler.instance = this;

        this.startTime = performance.now();
        this.memorySize = memorySize;
        this.pendingSize = pendingSize;

        this.renderPromise = null;
        this.isRendering = false;
        this.isWebcamInitialized = false;
        this.replayDataList = [];

        this.nextLabel = UNLABELED_IDX;
        this.lastRender = performance.now();

        this.updatesPerTimestep = 1;
        this.isStreamOn = false;
        this.trainOnNewData = true;

        this.similiarityGridFPS = new FPSCounter("SimilarityGrid FPS");

        // FOR DEBUG PURPOSES
        // Periodically log the number of tensors
        this.numTensorLogger = new VisLogger({
            name: "Number of Tensors over Time",
            tab: "Debug",
            xLabel: "Time",
            yLabel: "Number of Tensors",
        });
        setInterval(() => {
            const elapsed = Math.floor((performance.now() - this.startTime)/1000);
            this.numTensorLogger.push({x: elapsed, y: tf.memory().numTensors});
        }, 1000);
    }

    async initialize() {
        const inData = await loadImage("demo-pretrain-data/0/image1.png");
        tf.tidy(() => {
            const outData = modelHandler.model.predict(inData.Tensor);
            const dataEntry = new DataEntry(inData, outData, UNLABELED_IDX);
            this.cachedDataEntry = dataEntry;
        });
    }

    async reinitializeModel() {
        // Read the current value from the select element
        const architecture = d3.select("#arch-select").node().value;

        // Stop new data from being added
        this.stopRenderLoop();

        // Re-initialize the model
        // This is used when the architecture is changed
        await modelHandler.initializeModel({architecture: architecture});

        if (this.isStreamOn) {
            // Re-compute features for all data entries
            dataHandler.recomputeFeatures(modelHandler.model);

            // Re-render the datacards and the similarities
            await similarityGridHandler.renderDataCards();
            await similarityGridHandler.renderSimilarities();
        }
        // Allow new data to be added
        this.startRenderLoop();


    }

    async updateMemorySize(newSize) {
        if (newSize === this.memorySize) {
            return;
        }

        // Stop new data from being added
        this.stopRenderLoop();

        // Update the memory size
        this.memorySize = newSize;
        await dataHandler.updateMemoryAndPendingSize({memorySize: newSize});
        modelHandler.updateRandomIdxs();
        await similarityGridHandler.updateMemorySize(newSize);

        if (this.isStreamOn) {
            await new Promise(resolve => setTimeout(resolve, 500));
            // if the memory is not full, add card
            if (dataHandler.memoryEntries.length < newSize) {
                for (let i = dataHandler.memoryEntries.length; i < newSize; i++) {
                    // TODO: Wait for the user to add labels instead of auto-adding
                    // Reason for not implementing now is to avoid confusion on UI
                    await createDataCard(0);
                }
                // flush it out with unlabeled cards
                for (let i = 0; i < this.pendingSize; i++) {
                    await createDataCard(UNLABELED_IDX);
                }
            }

            // Update the similarity grid
            // dataHandler.updateSimilarities();
            await similarityGridHandler.renderSimilarities();
            // Allow new data to be added
        }
        this.startRenderLoop();

    }

    async updatePendingSize(newSize) {
        if (newSize === this.pendingSize) {
            return;
        }

        // Stop new data from being added
        this.stopRenderLoop();

        // Update the pending size
        this.pendingSize = newSize;
        await dataHandler.updateMemoryAndPendingSize({pendingSize: newSize});
        await similarityGridHandler.updatePendingSize(newSize);

        if (this.isStreamOn) {
            // if the pending is not full, add card
            if (dataHandler.pendingEntries.length < newSize) {
                // await new Promise(resolve => setTimeout(resolve, 500));
                for (let i = dataHandler.pendingEntries.length; i < newSize; i++) {
                    await createDataCard(UNLABELED_IDX);
                }
            }

            // Update the similarity grid
            dataHandler.updateSimilarities();
            await similarityGridHandler.renderSimilarities();
        }

        // Allow new data to be added
        this.startRenderLoop();
    }

    changeRandomSeed() {
        const seed = parseInt(d3.select("#random-seed-input").node().value);
        modelHandler.changeRandomSeed(seed);
    }

    changeOptimizer() {
        // Read the current value from the select element
        const optimizerString = d3.select("#optimizer-select").node().value;

        const learningRate = parseFloat(d3.select("#learning-rate-select").node().value);

        let optimizer;
        if (optimizerString === "momentum") {
            optimizer = tf.train.momentum(
                learningRate, 0.9
            );
        } else {
            optimizer = {
                "adam": tf.train.adam,
                "sgd": tf.train.sgd
            }[optimizerString](learningRate);
        }

        modelHandler.changeOptimizer(optimizer);
    }

    changeSelectionPolicy() {
        const policy1 = d3.select("#x1-policy-select").node().value;
        const policy2 = d3.select("#x2-policy-select").node().value;

        modelHandler.randomIdx = Math.floor(Math.random() * dataHandler.memorySize);
        modelHandler.randomIdx2 = Math.floor(Math.random() * dataHandler.memorySize);

        console.log(`randomIdx: ${modelHandler.randomIdx}`);
        console.log(`randomIdx2: ${modelHandler.randomIdx2}`);


        similarityGridHandler.X1Policy = policy1;
        similarityGridHandler.X2Policy = policy2;

        modelHandler.X1Policy = policy1;
        modelHandler.X2Policy = policy2;
    }

    changeFeatureUpdatePolicy() {
        const policy = d3.select("#features-select").node().value === true;
        modelHandler.updateFeatures = policy;
    }

    changeTrainOnNewData() {
        const trainOnNewData = d3.select("#train-on-new-select").node().value === "True";
        this.trainOnNewData = trainOnNewData;
    }

    changeUpdatesPerTimestep() {
        const updatesPerTimestep = parseInt(d3.select("#updates-per-timestep-select").node().value);
        // throw error if updatesPerTimestep is not between 1 and 5
        if (updatesPerTimestep < 1 || updatesPerTimestep > 5) {
            throw new Error("updatesPerTimestep must be between 1 and 5");
        }
        this.updatesPerTimestep = updatesPerTimestep;
    }

    updateSimilarityGridData() {
        // Assumes that all the datacard slots are filled already
        // This just shifts the underlying data of the datacards
        // following the same coreography as the animation
        // but only moving the content and not the DOM elements

        const newestDataEntry = dataHandler.currentEntry.clone();
        dataHandler.addDataEntry(newestDataEntry);
    }

    async updateSimilarityGridDOM() {
        this.similiarityGridFPS.update();
        // Update the DOM links
        const numMemoryEntries = dataHandler.memoryEntries.length;
        for (let i = 0; i < numMemoryEntries; i++) {
            similarityGridHandler.memoryCards[i].dataEntry = dataHandler.memoryEntries[i];
        }
        for (let i = 0; i < this.pendingSize; i++) {
            similarityGridHandler.pendingCards[i].dataEntry = dataHandler.pendingEntries[i];
        }

        return Promise.all([
            similarityGridHandler.renderDataCards(),
            similarityGridHandler.renderSimilarities(),
        ]);
    }
    async updateData() {
        if (this.renderPromise !== null) {
            return;
        }
        // Three cases:
        // 1 replayDataList is not empty - use it
        // 2 replayDataList is empty:
        //   2.1 webcam is initialized - capture webcam
        //   2.2 webcam is not initialized - use cached data

        // Throw error if the model is not initialized
        if (modelHandler.model === null) {
            throw new Error("Model is not initialized");
        }

        let dataEntry;
        if (this.replayDataList.length > 0) {
            const replayData = this.replayDataList.shift();
            // log the replay data
            const {label, dataURL} = replayData;
            const inData = await loadImage(dataURL);
            dataEntry = tf.tidy(() => {
                    const outData = modelHandler.model.predict(inData.Tensor);
                    return new DataEntry(inData, outData, label);
            });
        } else {
            dataEntry = tf.tidy(() => {
                if (this.isWebcamInitialized) {
                    const inData = captureWebcam();
                    const outData = modelHandler.model.predict(inData.Tensor);
                    return new DataEntry(inData, outData, this.nextLabel);
                }
                this.cachedDataEntry.label = this.nextLabel;
                return this.cachedDataEntry.clone();
            });
        }


        tf.tidy(() => {
            dataHandler.updateCurrentEntry(dataEntry);
            if (this.isStreamOn) {
                this.updateSimilarityGridData();
            }
        });
    }

    async updateDOM() {
        if (this.renderPromise !== null) {
            return;
        }
        this.renderPromise = this._updateDOM();
        await this.renderPromise;
        this.renderPromise = null;
    }

    _updateDOM() {
        const renderPromises = [predCardHandler.updateDOM()];

        if (this.isStreamOn) {
            renderPromises.push(this.updateSimilarityGridDOM());
        }
        return Promise.all(renderPromises);
    }

    keydown(event) {
        let label;
        switch (event.key.toLowerCase()) {
            case '1':
                label = 0;
                break;
            case '2':
                label = 1;
                break;
            case '3':
                label = 2;
                break;
            case 't':
                this.toggleTraining();
                return;
            default:
                label = UNLABELED_IDX;
                break;
        }
        this.nextLabel = label;

        // Make the button appear pressed using the ".active" css class
        const button = d3.select(`#addCategory${label}`);
        button.classed("active", true);
    }

    async keyup(event) {
        this.nextLabel = UNLABELED_IDX;
        // Remove the ".active" css class
        d3.selectAll(".addCategoryButton").classed("active", false);
    }

    async renderLoop() {
        if (!this.isRendering) {
            return;
        }

        if (performance.now() - this.lastRender > 1000 / REFRESH_RATE) {
            this.lastRender = performance.now();
            await this.updateData();
            await this.updateDOM();
        }
        window.requestAnimationFrame(() => {
            this.renderLoop();
        });
    }

    async initializeWebcam() {
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('aux-canvas');

        // Attempt to play video automatically
        video.setAttribute('autoplay', 'true');
        // Mute the video to allow autoplay without user interaction
        video.setAttribute('muted', 'true');
        // This attribute is important for autoplay in iOS
        video.setAttribute('playsinline', 'true');

        if (navigator.mediaDevices.getUserMedia) {
            video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true })
        }

        // Add event listener to the webcam feed
        video.addEventListener('loadeddata', () => {
        });

        // wait until the video is loaded
        await new Promise(resolve => {
            video.addEventListener('loadeddata', () => {
                console.log('Video loaded');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                resolve();
            });
        });

        this.isWebcamInitialized = true;
        const startStreamButton = document.getElementById("startStream");
        startStreamButton.disabled = false;

        const loadStreamButton = document.getElementById("loadStream");
        loadStreamButton.disabled = true;
    }

    async initializeModel() {
        await modelHandler.initializeModel({architecture: ARCHITECTURE, optimizer: OPTIMIZER});

        dataHandler.onMemoryFull = () => {
            modelHandler.evaluateModel();
            if (this.trainOnNewData) {
                this.trainModel();
            }
        }

        const webcamButton = document.getElementById("initializeWebcam");
        webcamButton.disabled = false;
        webcamButton.textContent = "Enable Webcam";

        const loadStreamButton = document.getElementById("loadStream");
        loadStreamButton.disabled = false;

    }

    async startStream() {
        if (this.isStreamOn) {
            return;
        }
        const streamButton = document.getElementById("startStream");
        streamButton.disabled = true;
        streamButton.textContent = "Adding Cards";
        await fillEmptySlots();
        streamButton.textContent = "Stream Started";
        this.isStreamOn = true;

        const saveStreamButton = document.getElementById("saveStream");
        saveStreamButton.disabled = false;
    }

    startRenderLoop() {
        if (this.isRendering) {
            return;
        }
        this.isRendering = true;
        this.renderLoop();
    }

    stopRenderLoop() {
        this.isRendering = false;
        // this will interrupt the render loop
    }

    trainModel() {
        console.log(`Training model for ${this.updatesPerTimestep} updates`);
        for (let i = 0; i < this.updatesPerTimestep; i++) {
            console.log(`Training model ${i}`);
            modelHandler.trainModel();
        }
    }
}
const eventHandler = new EventHandler();

// FOR SAVING
function saveAllImagesAsZip() {
    const zip = new JSZip();
    const labeledImages = dataHandler.labeledImages;

    for (let i = 0; i < labeledImages.length; i++) {
        const label = labeledImages[i].label;
        const dataURL = labeledImages[i].dataURL;
        const imageData = dataURL.split(',')[1];

        // Format the filename with a sequence number and label
        const filename = `${String(i).padStart(6, '0')}_label${label}.jpg`;

        // Add the image to the ZIP file
        zip.file(filename, imageData, { base64: true });
    }

    // Generate the ZIP file and trigger download
    zip.generateAsync({ type: 'blob' }).then((content) => {
        const link = document.createElement('a');
        link.href = URL.createObjectURL(content);
        link.download = 'labeled_images.zip';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
}

async function loadImagesFromZip(event) {
    const file = event.target.files[0];
    if (!file) {
        console.log('No file selected');
        return;
    }

    const zip = new JSZip();
    const content = await file.arrayBuffer();

    const zipContent = await zip.loadAsync(content);
    const images = [];

    // Get all filenames and sort them to maintain sequence
    const filenames = Object.keys(zipContent.files).sort();

    for (let filename of filenames) {
        const fileData = zipContent.files[filename];

        // Extract the label from the filename
        const labelMatch = filename.match(/label(\d+)/);
        const label = labelMatch ? parseInt(labelMatch[1]) : UNLABELED_IDX;

        // Read the image data as a data URL
        const dataURL = await fileData.async('base64').then((base64) => {
            return 'data:image/jpeg;base64,' + base64;
        });

        images.push({ dataURL, label });
    }


    // Now, images[] contains all the images in sequence.

    // Replay the images
    replayLabeledImages(images);
}


async function replayLabeledImages(images) {
    const numDataCards = eventHandler.pendingSize + eventHandler.memorySize;
    for (let i = 0; i < numDataCards; i++) {
        const { label, dataURL } = images.shift();
        await createDataCard(label, dataURL);

        // sleep for 50ms
        await new Promise(resolve => setTimeout(resolve, 50));
    }

    // Now the slots are filled, start the stream
    // and let the event handler consume the remaining images
    eventHandler.isStreamOn = true;
    eventHandler.replayDataList = images;

    const streamButton = document.getElementById("startStream");
    streamButton.textContent = "Stream Started";

    const saveStreamButton = document.getElementById("saveStream");
    saveStreamButton.disabled = false;
}

async function fillEmptySlots() {
    // Check if the memory is full
    if (dataHandler.memoryEntries.length === eventHandler.memorySize) {
        return;
    }

    // Fill the pending slots with webcam images
    for (let i = 0; i < eventHandler.memorySize; i++) {
        await createDataCard(0);
    }
    for (let i = 0; i < eventHandler.pendingSize; i++) {
        await createDataCard(UNLABELED_IDX);
    }
    modelHandler.randomIdx = Math.floor(Math.random() * dataHandler.memorySize);
    modelHandler.randomIdx2 = Math.floor(Math.random() * dataHandler.memorySize);

}


// Singleton elements
let dataHandler;
let modelHandler;
let similarityGridHandler;
let predCardHandler;


function initializeBackend() {
    // Initialize dataHandler
    dataHandler = new DataHandler({memorySize: MEMORY_SIZE, pendingSize: PENDING_SIZE});
    modelHandler = new ModelHandler();
}


async function initializeFrontend() {
    predCardHandler = new PredCardHandler();
    similarityGridHandler = new SimilarityGridHandler(
        {memorySize: MEMORY_SIZE, pendingSize: PENDING_SIZE}
    );

    // Connect the buttons to the event handler
    document.getElementById("initializeWebcam").
        addEventListener("click", async () => eventHandler.initializeWebcam());

    document.getElementById("startStream")
        .addEventListener("click", () => eventHandler.startStream());

    for (let i = 0 ; i < 3; i++) {
        button = document.getElementById(`addCategory${i}`);
        if (isMobile) {
            // on mobile the touchstart and touchend events are used
            button.addEventListener("touchstart", () => {
                eventHandler.nextLabel = i;
            });
            button.addEventListener("touchend", () => {
                eventHandler.nextLabel = UNLABELED_IDX;
            });
        } else {
            // on holding the button, the label is set to i
            button.addEventListener("mousedown", () => {
                eventHandler.nextLabel = i;
            });
            button.addEventListener("mouseup", () => {
                eventHandler.nextLabel = UNLABELED_IDX;
            });
        }
    }

    // Connect the UI to the event handlers
    const slider = document.getElementById("fpsSlider");
    slider.addEventListener("input", () => {
        REFRESH_RATE = parseInt(slider.value);
    });

    document.getElementById("trainModel")
        .addEventListener("click", () => eventHandler.trainModel());

    // set the default value
    memSizeInput = document.getElementById("memory-size-input")
    memSizeInput.value = MEMORY_SIZE;
    memSizeInput.addEventListener("change", () => {
            // check if the value is in the correct range
            const min = parseInt(memSizeInput.min);
            const max = parseInt(memSizeInput.max);
            let value = parseInt(memSizeInput.value);

            value = Math.min(max, Math.max(min, value));
            memSizeInput.value = value;
            eventHandler.updateMemorySize(value);
        })

    pendingSizeInput = document.getElementById("pending-size-input")
    pendingSizeInput.value = PENDING_SIZE;
    pendingSizeInput.addEventListener("change", () => {
            // check if the value is in the correct range
            const min = parseInt(pendingSizeInput.min);
            const max = parseInt(pendingSizeInput.max);
            let value = parseInt(pendingSizeInput.value);

            value = Math.min(max, Math.max(min, value));
            pendingSizeInput.value = value;
            eventHandler.updatePendingSize(value);
        });

    // First column of the advanced settings UI
    document.getElementById("arch-select")
        .addEventListener("change", () => eventHandler.reinitializeModel());
    document.getElementById("optimizer-select")
        .addEventListener("change", () => eventHandler.changeOptimizer());
    document.getElementById("learning-rate-select")
        .addEventListener("change", () => eventHandler.changeOptimizer());
    document.getElementById("updates-per-timestep-select")
        .addEventListener("change", () => eventHandler.changeUpdatesPerTimestep());

    // Second column of the advanced settings UI
    document.getElementById("x1-policy-select")
        .addEventListener("change", () => eventHandler.changeSelectionPolicy());
    document.getElementById("x2-policy-select")
        .addEventListener("change", () => eventHandler.changeSelectionPolicy());
    document.getElementById("features-select")
        .addEventListener("change", () => eventHandler.changeFeatureUpdatePolicy());
    document.getElementById('train-on-new-select')
        .addEventListener('change', () => modelHandler.updateRandomIdxs());


    document.getElementById('saveStream').addEventListener('click', saveAllImagesAsZip);

    document.getElementById('loadStream').addEventListener('click', function() {
        document.getElementById('loadStreamInput').click();
    });

    document.getElementById('loadStreamInput').addEventListener('change', loadImagesFromZip);

    await Promise.all([
        similarityGridHandler.initialize(),
        eventHandler.initializeModel(),
    ]);
    await eventHandler.initialize();
    eventHandler.startRenderLoop();
}


// Initialize the application on page load
document.addEventListener('DOMContentLoaded', async () => {
    initializeBackend();
    await initializeFrontend();
});

// stop the setInterval and webcam when the user switches tabs
document.addEventListener('visibilitychange', async () => {
    if (document.visibilityState === "hidden") {
        if (eventHandler.isRendering) {
            console.log("Stopping the render loop");
            eventHandler.stopRenderLoop();
        }
    } else if (document.visibilityState === "visible") {
        if (!eventHandler.isRendering) {
            console.log("Starting the render loop");
            eventHandler.startRenderLoop();
        }
    }
});

document.addEventListener('keydown', async (event) => {
    eventHandler.keydown(event);
});

document.addEventListener('keyup', async (event) => {
    eventHandler.keyup(event);
});
