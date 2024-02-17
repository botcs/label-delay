const PENDING_SIZE = 18;
const MEMORY_SIZE = 40;
// const PENDING_SIZE = 5;
// const MEMORY_SIZE = 10;
const NUM_CLASSES = 3;
const NUM_FEATURES = 3**2;
const UNLABELED = 42;

const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
let REFRESH_RATE = 30;

////////////////////////////////////////
// Machine Learning params
////////////////////////////////////////
const LR = 0.0001;
// const MOMENTUM = 0.3;
const OPTIMIZER = tf.train.sgd(LR);
// const OPTIMIZER = tf.train.momentum(LR, MOMENTUM);
// const OPTIMIZER = tf.train.adam(LR);
const ARCHITECTURE = "mobilenetv3";
// const ARCHITECTURE = "cnn_base";
// const ARCHITECTURE = "resnet18";
// const ARCHITECTURE = "linear";
const IMAGE_SIZE = 32;
const IMAGE_CHANNELS = 3;
const TRAIN_REPEAT_INTERVAL = 2000;



const video = document.getElementById('webcam');

// Attempt to play video automatically
video.setAttribute('autoplay', 'true');
video.setAttribute('muted', 'true'); // Mute the video to allow autoplay without user interaction
video.setAttribute('playsinline', 'true'); // This attribute is important for autoplay in iOS

const canvas = document.getElementById('aux-canvas');
const mainSvg = d3.select("#similarity-grid");
const predSvg = d3.select("#prediction-card");

// Define the clipping path
const clip = mainSvg.append("defs")
    .append("clipPath")
    .attr("id", "clip-rounded-rect")
    .append("rect")

////////////////////////////////////////
// BACKEND
////////////////////////////////////////
class FPSCounter {
    constructor(name, warmup=5000, periodicLog=5000) {
        this.frames = 0;
        this.lastFPS = -1;
        this.startTime = performance.now();
        this.warmup = warmup;

        this.periodicLog = periodicLog;
        this.lastLog = performance.now();

        this.name = name;
    }

    update() {
        this.frames++;
        const currentTime = performance.now();
        const elapsedTime = currentTime - this.startTime;
        if (elapsedTime > this.warmup) {
            this.lastFPS = this.frames / (elapsedTime / 1000);
            this.startTime = currentTime;
            
            if (currentTime - this.lastLog > this.periodicLog) {
                this.lastLog = currentTime;
                this.log();
            }
            
            this.frames = 0;
        }
    }

    log() {
        console.log(`${this.name} - moving avg FPS: ${this.lastFPS.toFixed(2)}`);
    }
}


class DataEntry {
    static count = 0;
    constructor(inData, outData, label = UNLABELED) {
        this.inData = inData; // An image or a video frame

        this.inData.Tensor = tf.variable(inData.Tensor);

        // A [NUM_CLASES] tensor representing the class probabilities
        this.pred = tf.variable(tf.squeeze(tf.softmax(outData[0])));

        // A [NUM_FEATURES] tensor representing the feature vector
        const feat = outData[1];
        this.feat = tf.variable(tf.squeeze(feat.div(feat.norm(2))));
        this.label = label; // A string or number representing the label
        this.id = DataEntry.count;


        this.dataCard = null;
        DataEntry.count++;
    }

    dispose() {
        this.inData.Tensor.dispose();
        this.pred.dispose();
        this.feat.dispose();
    }

    updateData(inData, outData=null) {
        // Update the data of the entry
        this.inData.dataURL = inData.dataURL;
        if (outData === null) {
            return;
        }
        this.inData.Tensor.assign(inData.Tensor);
        const pred = tf.softmax(outData[0]);
        this.pred.assign(tf.squeeze(pred));
        const feat = outData[1];
        this.feat.assign(tf.squeeze(feat.div(feat.norm(2))));
    }

    clone() {
        return new DataEntry(this.inData, [this.pred, this.feat], this.label);
    }
}

class DataHandler {
    constructor(memorySize = MEMORY_SIZE, pendingSize = PENDING_SIZE) {
        this.memorySize = memorySize;
        this.pendingSize = pendingSize;

        this.currentEntry = null;

        this.memoryEntries = [];
        this.pendingEntries = [];


        // Use this to store the cosine similarities
        this.similarities = tf.variable(tf.zeros([this.pendingSize, this.memorySize]));


        // Use this to store softmax scores
        this.scores = tf.variable(tf.zeros([this.pendingSize, this.memorySize]));

        this.FPS = new FPSCounter("DataHandler");
    }

    updateCurrentEntry() {
        // Update the current entry
        tf.tidy(() => {
            const inData = captureWebcam();
            const outData = modelHandler.model.predict(inData.Tensor);
            if (this.currentEntry === null) {
                this.currentEntry = new DataEntry(inData, outData);
            } else {
                this.currentEntry.updateData(inData, outData);
            }
        });
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

    
    async addDataEntry(dataEntry) {
        this.FPS.update();
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
            if (transitionEntry.label !== UNLABELED) {
                this.memoryEntries.unshift(transitionEntry);

                // Add new column to the left and remove right column
                
                const simMemToPen = this.computeSingleSimilarity(
                    transitionEntry, this.pendingEntries, this.pendingSize
                );
                this.similarities.assign(this.unshiftAndPop(
                    this.similarities, simMemToPen, 1
                ));
            } else {
                transitionEntry.dispose();
            }
        }

        // If the capacity of the memory entries is exceeded
        if (this.memoryEntries.length > this.memorySize) {
            // Remove the oldest memory entry
            const oldestEntry = this.memoryEntries.pop();
            oldestEntry.dispose();
        }

        // Update the softmax scores
        this.scores.assign(tf.softmax(this.similarities, 1));
    }

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
        const padLength = this.memorySize - keyEntries.length;
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
const dataHandler = new DataHandler();



class ModelHandler{
    constructor(dataHandler) {
        this.model = null;
        this.dataHandler = dataHandler;
        this.seed = 42;
        this.prevLabel = null;
        this.randomIdx = Math.floor(Math.random() * dataHandler.memoryEntries.length);
        this.randomIdx2 = Math.floor(Math.random() * dataHandler.memoryEntries.length);

        this.X1Policy = "random";
        this.X2Policy = "random2";
        this.updateFeatures = true;
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
        this.numIterations = 0;

        similarityGridHandler.THETA_T.text(this.numIterations)
            .append("tspan")
            .attr("dy", "0.5em")
            .text("=train(");

    }

    async changeOptimizer(optimizer) {
        this.model.optimizer = optimizer;
    }

    async trainModel() {
        // Assert that at least one memory entry is present
        if (dataHandler.memoryEntries.length === 0) {
            throw new Error("At least one labeled data entry is required");
        }

        // Train the model
        const optimizer = this.model.optimizer;
        const lossFunction = tf.losses.softmaxCrossEntropy;
        
        // const indices = [];
        // // find memory samples that are not the same as the previous label
        // const availableIndices = [];
        // for (let i = 0; i < dataHandler.memoryEntries.length; i++) {
        //     if (this.prevLabel !== dataHandler.memoryEntries[i].label) {
        //         availableIndices.push(i);
        //     }
        // }
        
        // let randomIdx;
        // if (availableIndices.length === 0) {
        //     randomIdx = Math.floor(Math.random() * dataHandler.memoryEntries.length);
        // } else {
        //     randomIdx = availableIndices[Math.floor(Math.random() * availableIndices.length)];
        // }
        
        // this.prevLabel = dataHandler.memoryEntries[randomIdx].label;
        // console.log(`randomIdx: ${randomIdx}, prevLabel: ${this.prevLabel}`);
        // this.randomIdx = randomIdx;
        // indices.push(randomIdx);

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

        const useFrozenBackbone = this.model.architecture.includes("mobilenet");
        if ( useFrozenBackbone) {
            const features = this.model.forwardBackbone(data.input);
            // Just exclude the backbone from the optimization
            optimizer.minimize(() => {
                const logits = this.model.forwardHead(features)[0];
                const loss = lossFunction(data.labels, logits);
                console.log(`Training iteration: ${this.numIterations} - Loss: ${loss.dataSync()}`);
                return loss;
            });
        } else {
            optimizer.minimize(() => {
                const features = this.model.forwardBackbone(data.input);
                const logits = this.model.forwardHead(features)[0];
                const loss = lossFunction(data.labels, logits);
                console.log(`Training iteration: ${this.numIterations} - Loss: ${loss.dataSync()}`);
                return loss;
            });
        }
        data.input.dispose();
        data.labels.dispose();

        this.numIterations++;
        similarityGridHandler.THETA_T.text(this.numIterations)
            .append("tspan")
            .attr("dy", "0.5em")
            .text("=train(");


        if (this.updateFeatures) {
            // Update all the features
            tf.tidy(() => {
                dataHandler.recomputeFeatures(this.model);
            });
            // console.log(`Number of tensors: ${tf.memory().numTensors}`);
            
            similarityGridHandler.renderDataCards();
            similarityGridHandler.renderSimilarities();
        }
    }
}
const modelHandler = new ModelHandler(dataHandler);

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

    async createDOM(parentGroup = null) {
        if (this.renderPromise !== null) {
            return this.renderPromise;
        }
        this.renderPromise = this._createDOM(parentGroup);
        await this.renderPromise;
        this.renderPromise = null;
    }

    async _createDOM(parentGroup = null) {
        if (parentGroup === null) {
            parentGroup = mainSvg;
        }
        const mainGroup = parentGroup
            .append("g");
        
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

    async createDOM(parentGroup = null) {
        if (this.renderPromise !== null) {
            return this.renderPromise;
        }
        this.renderPromise = this._createDOM(parentGroup);
        await this.renderPromise;
        this.renderPromise = null;
    }

    async _createDOM(parentGroup = null) {
        if (parentGroup === null) {
            parentGroup = predSvg;
        }
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
    constructor(dataHandler) {
        this.dataHandler = dataHandler;
        this.predCard = null;
        this.dataCard = null;
        this.renderPromise = null;
    }

    async createDOM() {
        const offset = {x: 5, y: 5};

        this.mainGroup = predSvg.append("g");

        this.dataCard = new DataCard(
            this.dataHandler.pendingEntries[0], 
            {x: offset.x, y: offset.y},
            "horizontal"
        );
        await this.dataCard.createDOM(this.mainGroup);
        
        this.predCard = new PredCard(
            this.dataHandler.pendingEntries[0], 
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
    }

    async updateDOM() {
        this.dataCard.dataEntry = this.dataHandler.pendingEntries[0];
        this.predCard.dataEntry = this.dataHandler.pendingEntries[0];
        await this.dataCard.updateDOM();
        await this.predCard.updateDOM();
    }
}
const predCardHandler = new PredCardHandler(dataHandler);


class SimilarityGridHandler {
    constructor(
        memorySize = MEMORY_SIZE, 
        pendingSize = PENDING_SIZE, 
        offset = {x:0, y:0},
        // boardWidth = 1600, 
        // boardHeight = 500
    ) {
        // the memory entries are the labeled datacards
        // the pending entries are the unlabeled datacards
        // the width and height are used for the inner dimensions of table

        this.memorySize = memorySize;
        this.pendingSize = pendingSize;
        this.memoryCards = [];
        this.pendingCards = [];
        this.offset = offset;

        // width is 1x[horizontal card] + (MEM+1)x[vertical card]
        this.boardWidth = DataCard.layouts["horizontal"].shape.width;
        this.boardWidth += (this.memorySize + 1) * DataCard.layouts["vertical"].shape.width;

        // height is PENDx[horizontal card] + 1x[vertical card]
        this.boardHeight = this.pendingSize * DataCard.layouts["horizontal"].shape.height;
        this.boardHeight += DataCard.layouts["vertical"].shape.height;

        this.equationHeight = DataCard.unitSize*2;

        // set the viewbox of the svg
        mainSvg.attr("viewBox", `0 0 ${this.boardWidth} ${this.boardHeight + this.equationHeight}`);
        this.renderPromise = null;

        this.X1Policy = "random";
        this.X2Policy = "random2";
    }

    async initialize() {
        this.setDOMPositions();
        this.mainGroup = mainSvg.append("g")
            .attr("id", "DOMHandler")
            .attr("transform", `translate(${this.offset.x}, ${this.offset.y})`);
        
        // draw the grid
        const grid = this.mainGroup.append("g")
        grid.append("g")
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

        grid.append("g")
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


        this.legendGroup = this.mainGroup.append("g")
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

        

        this.similarityGroup = this.mainGroup.append("g")
            .attr("id", "similarityGroup");

        this.THETA_T = this.mainGroup.append("text")
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

        this.X_1 = this.mainGroup.append("text")
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
        this.X_2 = this.mainGroup.append("text")
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
            // .style("stroke", "#797979")
            // .style("stroke-width", "2px")
            .attr("rx", 10)
            .attr("ry", 10);
        
        // Draw arrows from max similarity to the equation
        if (this.arrowX1 === undefined) {
            this.arrowX1 = mainSvg.append("path")
                .attr("stroke", "red")
                .attr("stroke-width", 6)
                .attr("fill", "none");
        }
        if (this.arrowX2 === undefined) {
            this.arrowX2 = mainSvg.append("path")
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
        .duration(150)
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
        await dataCard.createDOM(this.mainGroup);

        this.pendingCards.unshift(dataCard);
        
        let transitionCard = null
        if (this.pendingCards.length > this.pendingSize) {
            transitionCard = this.pendingCards.pop();
        }
        const asyncMoveCall = this.updatePendingPositions();

            
        if (transitionCard !== null) {
            // if labeled, add to the memory entries
            if (transitionCard.dataEntry.label !== UNLABELED) {
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
        if (this.memoryCards.length > 0) {
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
const similarityGridHandler = new SimilarityGridHandler();


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
    const ctx = canvas.getContext('2d');

    const thumbnailSize = d3.max([
        // similarityGridHandler.getThumbnailClientSize().width, 
        224
    ]) * 2;

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

async function createDataCard(label = UNLABELED, url = null) {
    if (similarityGridHandler.renderPromise !== null) {
        await similarityGridHandler.renderPromise;
        
    }
    let inData;
    if (url !== null) {
        inData = await loadImage(url);
    }

    // outData = model.predict(inData);
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


// async function updatePendingDataCard(idx) {
//     // measure the time it takes to update the pending datacard
//     const start = performance.now();
    
//     if (similarityGridHandler.pendingCards.length === 0) {
//         return;
//     }

//     // if the user is using a mobile device limit the fps to 2
//     // I could use a counter, but then async headaches...
//     const skipInference = isMobile && Math.random() > 2/REFRESH_RATE;
//     // const skipInference = true;

//     // Update the newest data entry
//     const dataEntry = dataHandler.pendingEntries[idx];
//     tf.tidy(() => {
//         const inData = captureWebcam();
//         if (skipInference) {
//             dataEntry.updateData(inData, null);
//             return;
//         }
//         const outData = modelHandler.model.predict(inData.Tensor);
//         dataEntry.updateData(inData, outData);
//         dataHandler.updateSimilaritiesRow(idx);
//     });
//     image_hash = await dataHandler.pendingEntries[idx].inData.Tensor.data();
//     // hash image to compare with the previous image

//     await similarityGridHandler.renderSimilarities();
//     await similarityGridHandler.pendingCards[idx].updateDOM();
//     const end = performance.now();

//     // benchmark the time it takes to update the pending datacard
//     // console.log(`Updating the pending datacard took ${end-start} ms`);
// }

class EventHandler {
    constructor() {
        this.renderPromise = null;
        this.isRendering = true;
        this.isWebcamInitialized = false;

        this.nextLabel = UNLABELED;
        this.lastRender = performance.now();

        this.trainInterval = null;

    }

    async reinitializeModel() {
        // Read the current value from the select element
        const architecture = d3.select("#arch-select").node().value;

        // Stop new data from being added
        this.stopRenderLoop();

        // Re-initialize the model
        // This is used when the architecture is changed
        await modelHandler.initializeModel({architecture: architecture});

        // Re-compute features for all data entries
        await dataHandler.recomputeFeatures(modelHandler.model);

        // Re-render the datacards and the similarities
        await similarityGridHandler.renderDataCards();
        await similarityGridHandler.renderSimilarities();

        // Allow new data to be added
        this.startRenderLoop();
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
        const policy = d3.select("#features-select").node().value;
        modelHandler.updateFeatures = policy === true;
    }

        

    async addDataWithoutAnimation(label=UNLABELED) {
        if (this.renderPromise !== null) {
            return this.renderPromise;
        }
        this.renderPromise = this._addDataWithoutAnimation(label);
        await this.renderPromise;
        this.renderPromise = null;
    }

    async _addDataWithoutAnimation(label=UNLABELED) {
        // Assumes that all the datacard slots are filled already
        // This just shifts the underlying data of the datacards
        // following the same coreography as the animation
        // but only moving the content and not the DOM elements
        tf.tidy(() => {
            const inData = captureWebcam();
            const outData = modelHandler.model.predict(inData.Tensor);
            const dataEntry = new DataEntry(inData, outData, label);
            
            // this handles all the shifting
            dataHandler.addDataEntry(dataEntry);
        });

        // Update the data entries
        // const newestDataEntry = dataHandler.currentEntry.clone();
        // newestDataEntry.label = label;
        // dataHandler.addDataEntry(newestDataEntry);

        // Update the DOM links
        for (let i = 0; i < PENDING_SIZE; i++) {
            similarityGridHandler.pendingCards[i].dataEntry = dataHandler.pendingEntries[i];
        }
        for (let i = 0; i < MEMORY_SIZE; i++) {
            similarityGridHandler.memoryCards[i].dataEntry = dataHandler.memoryEntries[i];
        }


        
        await Promise.all([
            similarityGridHandler.renderDataCards(),
            similarityGridHandler.renderSimilarities(),
        ]);
    }

    async keydown(event) {
        let label;
        switch (event.key.toLowerCase()) {
            case '1':
                label = '0';
                break;
            case '2':
                label = '1';
                break;
            case '3':
                label = '2';
                break;
            case 't':
                this.toggleTraining();
                return;
            default:
                label = UNLABELED;
                break;
        }
        this.nextLabel = label;
    }

    async keyup(event) {
        this.nextLabel = UNLABELED;
    }

    async renderLoop() {
        if (!this.isRendering) {
            return;
        }
        if (performance.now() - this.lastRender > 1000 / REFRESH_RATE) {   
            this.lastRender = performance.now();
            // dataHandler.updateCurrentEntry();
            this.addDataWithoutAnimation(this.nextLabel);
            predCardHandler.updateDOM();
        }
        window.requestAnimationFrame(() => {
            this.renderLoop();
        });
    }

    async initializeWebcam() {
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
    }

    startRenderLoop() {
        this.isRendering = true;
        this.renderLoop();
    }

    stopRenderLoop() {
        this.isRendering = false;
        // this will interrupt the render loop
    }

    toggleTraining() {
        const button = d3.select("#trainModel");
        
        if (this.trainInterval === undefined) {
            this.trainInterval = null;
        }

        if (this.trainInterval !== null) {
            clearInterval(this.trainInterval);
            button.style("background", `white`)
                .text("Toggle Training")
                .transition()
                .duration(TRAIN_REPEAT_INTERVAL)
                .style("background", `white`);
            this.trainInterval = null;
        } else {
            button.text("Stop Training")
            let transition = button.style("background", `linear-gradient(90deg, #6aa84f 0%, white 1%)`);
            for (let i = 0; i < 30; i++) {
                const percent = Math.floor(i * 100 / 30);
                transition = transition.transition()
                    .duration(TRAIN_REPEAT_INTERVAL / 30)
                    .style("background", `linear-gradient(90deg, #6aa84f ${percent}%, white ${percent+1}%)`);
            }

            this.trainInterval = setInterval(async () => {

                // The actual training happens here
                // The rest is just frontend updates
                modelHandler.trainModel();

                let transition = button.style("background", `linear-gradient(90deg, #6aa84f 0%, white 0%)`);
                for (let i = 0; i < 30; i++) {
                    const percent = Math.floor(i * 100 / 30);
                    transition = await transition.transition()
                        .duration(TRAIN_REPEAT_INTERVAL / 30)
                        .style("background", `linear-gradient(90deg, #6aa84f ${percent}%, white ${percent}%)`);
                }

            }, TRAIN_REPEAT_INTERVAL);
        }
    }
}
const eventHandler = new EventHandler();


async function fillSlots(useWebcam=true) {
    if (useWebcam) {
        // Fill the pending slots with webcam images
        for (let i = 0; i < PENDING_SIZE+MEMORY_SIZE; i++) {
            await createDataCard(0);
        }
        modelHandler.randomIdx = Math.floor(Math.random() * dataHandler.memorySize);
        modelHandler.randomIdx2 = Math.floor(Math.random() * dataHandler.memorySize);
    } else {
        // Randomly select images from the dataset to fill all slot
        const data = [];
        const numData = PENDING_SIZE + MEMORY_SIZE;
        for (let i = 0; i < numData; i++) {
            const label = Math.floor(Math.random() * 2);
            const url = `demo-pretrain-data/${label}/image${i%5}.png`;
            await createDataCard(label, url);
        }
    }
}

function downloadAllImagesAsZip() {
    const zip = new JSZip();
    const categoryIndices = {};

    dataHandler.memoryEntries.forEach((dataEntry) => {
        const category = dataEntry.label;
        // Initialize the category index if not already done
        if (!(category in categoryIndices)) {
            categoryIndices[category] = 0;
        }

        const filename = `image${categoryIndices[category]}.jpg`;
        // Assuming dataURL is in the format "data:image/jpg;base64,..."
        const imageData = dataEntry.inData.dataURL.split(',')[1];

        // Create a folder for the category if it doesn't exist
        if (!zip.folder(category)) {
            zip.folder(category);
        }

        // Add the image to the appropriate folder
        zip.folder(category).file(filename, imageData, { base64: true });

        // Increment the index for this category
        categoryIndices[category]++;
    });

    zip.generateAsync({ type: 'blob' }).then((content) => {
        const link = document.createElement('a');
        link.href = URL.createObjectURL(content);
        link.download = 'images.zip';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
}

// Initialize the webcam on page load
document.addEventListener('DOMContentLoaded', 
    async () => {
        // Connect the buttons
        for (let i = 0 ; i < 3; i++) {
            button = document.getElementById(`addCategory${i}`);
            if (isMobile) {
                // on mobile the touchstart and touchend events are used
                button.addEventListener("touchstart", () => {
                    eventHandler.nextLabel = i.toString();
                });
                button.addEventListener("touchend", () => {
                    eventHandler.nextLabel = UNLABELED;
                });
            } else {
                // on holding the button, the label is set to i
                button.addEventListener("mousedown", () => {
                    eventHandler.nextLabel = i.toString();
                });
                button.addEventListener("mouseup", () => {
                    eventHandler.nextLabel = UNLABELED;
                });
            }

        }

        // Connect the slider
        const slider = document.getElementById("fpsSlider");
        slider.addEventListener("input", () => {
            REFRESH_RATE = parseInt(slider.value);
        });


        document.getElementById("trainModel")
            .addEventListener("click", () => eventHandler.toggleTraining());
        document.getElementById("saveImages")
            .addEventListener("click", downloadAllImagesAsZip);


        document.getElementById("arch-select")
            .addEventListener("change", () => eventHandler.reinitializeModel());
        document.getElementById("optimizer-select")
            .addEventListener("change", () => eventHandler.changeOptimizer());
        document.getElementById("learning-rate-select")
            .addEventListener("change", () => eventHandler.changeOptimizer());
        document.getElementById("x1-policy-select")
            .addEventListener("change", () => eventHandler.changeSelectionPolicy());
        document.getElementById("x2-policy-select")
            .addEventListener("change", () => eventHandler.changeSelectionPolicy());
        document.getElementById("features-select")
            .addEventListener("change", () => eventHandler.changeFeatureUpdatePolicy());
            

        await Promise.all([
            similarityGridHandler.initialize(),
            modelHandler.initializeModel(),
            eventHandler.initializeWebcam(),
        ]);

        await createDataCard();
        await fillSlots();
        await predCardHandler.createDOM();
        
        eventHandler.renderLoop();
    }
);

// stop the setInterval and webcam when the user switches tabs
document.addEventListener('visibilitychange', async () => {
    if (document.visibilityState === "hidden") {
        if (eventHandler.isRendering) {
            console.log("Stopping the render loop");
            eventHandler.stopRenderLoop();

            if (eventHandler.trainInterval !== null) {
                console.log("Stopping the training interval");
                eventHandler.toggleTraining();
            }
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
