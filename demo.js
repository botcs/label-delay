const PENDING_SIZE = 3;
const MEMORY_SIZE = 13;
const NUM_CLASSES = 3;
const NUM_FEATURES = 3**2;
const UNLABELED = 42;

////////////////////////////////////////
// Machine Learning params
////////////////////////////////////////
const LR = 0.005;
const MOMENTUM = 0.0;
const OPTIMIZER = tf.train.sgd(LR);
// const OPTIMIZER = tf.train.momentum(LR, MOMENTUM);
// const OPTIMIZER = tf.train.adam(LR);
const IMAGE_SIZE = 32;
const IMAGE_CHANNELS = 3;
const ARCHITECTURE = "cnn";



const video = document.getElementById('webcam');
const canvas = document.getElementById('aux-canvas');
const svg = d3.select("#main")

// Define the clipping path
const clip = svg.append("defs")
    .append("clipPath")
    .attr("id", "clip-rounded-rect")
    .append("rect")

////////////////////////////////////////
// BACKEND
////////////////////////////////////////
class DataEntry {
    static count = 0;
    constructor(inData, outData, label = UNLABELED) {
        this.inData = inData; // An image or a video frame

        this.inData.Tensor = tf.variable(inData.Tensor);
        // A [NUM_CLASES] tensor representing the class probabilities
        this.pred = tf.variable(tf.squeeze(outData[0]));

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

    updateData(inData, outData) {
        // Update the data of the entry
        this.inData.dataURL = inData.dataURL;
        this.inData.Tensor.assign(inData.Tensor);
        this.pred.assign(tf.squeeze(outData[0]));
        const feat = outData[1];
        this.feat.assign(tf.squeeze(feat.div(feat.norm(2))));
    }
}

class DataHandler {
    constructor(memorySize = MEMORY_SIZE, pendingSize = PENDING_SIZE) {
        this.memorySize = memorySize;
        this.pendingSize = pendingSize;

        this.memoryEntries = [];
        this.pendingEntries = [];


        // Use this to store the cosine similarities
        this.similarities = tf.variable(tf.zeros([this.pendingSize, this.memorySize]));


        // Use this to store softmax scores
        this.scores = tf.variable(tf.zeros([this.pendingSize, this.memorySize]));
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



class Trainer{
    constructor(dataHandler) {
        this.model = null;
        this.dataHandler = dataHandler;
        this.numIterations = 0;
        this.seed = 42;
        this.prevLabel = null;
        this.randomIdx = Math.floor(Math.random() * dataHandler.memoryEntries.length);
    }
    

    async initializeModel() {
        Math.seedrandom('constant');
        const backboneFactory = {
            "resnet18": resNet18,
            "mobilenetv2": mobileNetV2,
            "cnn": CNN
        }[ARCHITECTURE];
        const model = tf.tidy(() => {
            const backbone = backboneFactory([IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS]);
            const input = backbone.input
            const proj = tf.layers.dense({
                units: NUM_FEATURES,
                activation: "tanh",
                kernelInitializer: "varianceScaling",
                kernelRegularizer: "l1l2",
            }).apply(backbone.output);
            const logit = tf.layers.dense({units: NUM_CLASSES}).apply(proj);
            
            return tf.model({
                inputs: input, 
                outputs: [logit, proj]
            });
        });

        
        
        
        // const optimizer = tf.train.sgd(0.005);
        const noopLoss = (yTrue, yPred) => tf.zeros([1]);
        await model.compile({
            optimizer: OPTIMIZER,
            // loss: {
            //     pred: 'categoricalCrossentropy',
            //     feat: noopLoss
            // },
            loss: ['categoricalCrossentropy', noopLoss],
            metrics: ['accuracy']
        });

        // Warm up the model by training it once
        const warmupData = tf.randomNormal([6, 32, 32, 3]);
        const labels = tf.oneHot(tf.tensor1d([0,0,1,1,2,2], 'int32'), NUM_CLASSES);
        let logits;
        let loss;
        await OPTIMIZER.minimize(() => {
            logits = model.predict(warmupData)[0];
            loss = tf.losses.softmaxCrossEntropy(labels, logits).mul(0);
            return loss;
        });

        // clean up the tensors
        warmupData.dispose();
        logits.dispose();
        labels.dispose();
        loss.dispose();

        // Summary
        console.log(model.summary());
        this.model = model;
    }

    async trainModel() {
        // Add an unlabeled entry to the pending entries
        await createDataCard(UNLABELED);

        // Train the model
        const optimizer = this.model.optimizer;
        const lossFunction = tf.losses.softmaxCrossEntropy;
        const batchSize = 2;

        const indices = [];
        // for (let i = 0; i < batchSize; i++) {
        //     const idx = Math.floor(Math.random() * dataHandler.memoryEntries.length);
        //     indices.push(idx);
        // }
        // let randomIdx = Math.floor(Math.random() * dataHandler.memoryEntries.length);
        // while (this.prevLabel === dataHandler.memoryEntries[randomIdx].label) {
        //     randomIdx = Math.floor(Math.random() * dataHandler.memoryEntries.length);
        // }

        // find memory samples that are not the same as the previous label
        const availableIndices = [];
        for (let i = 0; i < dataHandler.memoryEntries.length; i++) {
            if (this.prevLabel !== dataHandler.memoryEntries[i].label) {
                availableIndices.push(i);
            }
        }
        const randomIdx = availableIndices[Math.floor(Math.random() * availableIndices.length)];
        this.prevLabel = dataHandler.memoryEntries[randomIdx].label;
        this.randomIdx = randomIdx;
        indices.push(randomIdx);
        

        // Read the IWM index from DataHandler
        const iwmIdx = dataHandler.scores.argMax(1).dataSync()[0];
        console.log(`randomIdx: ${randomIdx}, iwmIdx: ${iwmIdx}`);
        indices.push(iwmIdx);
        
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
        optimizer.minimize(() => {
            const logits = this.model.predict(data.input)[0];
            const loss = lossFunction(data.labels, logits);
            loss.print();
            return loss;
        });
        data.input.dispose();
        data.labels.dispose();

        this.numIterations++;
        domHandler.THETA_T.text(this.numIterations)
            .append("tspan")
            .attr("dy", "0.5em")
            .text("=train(");
        // d3.selectAll("#theta-t").text(this.numIterations);

        // Update all the features
        tf.tidy(() => {
            dataHandler.recomputeFeatures(this.model);
        });
        await domHandler.renderDataCards();
        await domHandler.renderSimilarities();
    }
}
const trainer = new Trainer(dataHandler);

////////////////////////////////////////
// FRONTEND
////////////////////////////////////////
class DataCard {
    // A datacard is a visual representation of a data entry
    static unitSize = 100;

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
            parentGroup = svg;
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


    async changeOrientation(orientation, duration=250) {
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
            .duration(duration)
            .attr("width", diagonalLayout.shape.width)
            .attr("height", diagonalLayout.shape.height);

        this.imageGroup.transition()
            .duration(duration)
            .attr("transform", `translate(${diagonalLayout.image.x}, ${diagonalLayout.image.y})`);
            

        await this.featureGroup.transition()
            .duration(duration)
            .attr("transform", `translate(${diagonalLayout.feature.x}, ${diagonalLayout.feature.y})`)
            .end();

        this.layout = DataCard.layouts[orientation];
        this.orientation = orientation;
            
        // transition to diagonal
        this.background.transition()
            .duration(duration)
            .attr("width", this.layout.shape.width)
            .attr("height", this.layout.shape.height);
        
        this.imageGroup.transition()
            .duration(duration)
            .attr("transform", `translate(${this.layout.image.x}, ${this.layout.image.y})`);

        this.featureGroup.transition()
            .duration(duration)
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
            .duration(500)
            .attr(
                "transform", `translate(${x}, ${y})`
            )
            .end();
    }

    removeDOM() {
        // Remove the DOM elements
        // use fading out animation
        this.mainGroup.transition()
            .duration(500)
            .style("opacity", 0)
            .remove();
    }

}



class DOMHandler {
    constructor(
        memorySize = MEMORY_SIZE, 
        pendingSize = PENDING_SIZE, 
        offset = {x:0, y:0},
        boardWidth = 1500, 
        boardHeight = 500
    ) {
        // the memory entries are the labeled datacards
        // the pending entries are the unlabeled datacards
        // the width and height are used for the inner dimensions of table

        this.memorySize = memorySize;
        this.pendingSize = pendingSize;
        this.memoryCards = [];
        this.pendingCards = [];
        this.offset = offset;
        this.boardWidth = boardWidth;
        this.boardHeight = boardHeight;

        this.initialize();
        this.renderPromise = null;
    }

    async initialize() {
        this.setDOMPositions();
        this.mainGroup = svg.append("g")
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

        this.X_RND = this.mainGroup.append("text")
            .attr("id", "X_RND")
            .style("font-size", "3em")
            .style("text-anchor", "middle")
            .style("dominant-baseline", "hanging")
            .attr("x", this.boardWidth*0.5)
            .attr("y", this.boardHeight + DataCard.unitSize)
            .text("X");
        this.X_RND.append("tspan")
            .attr("dy", "0.5em")
            .text("RND")
            .append("tspan")
            .attr("dy", "-0.5em")
            .text(",");
        this.X_IWM = this.mainGroup.append("text")
            .attr("id", "X_IWM")
            .style("font-size", "3em")
            .style("text-anchor", "middle")
            .style("dominant-baseline", "hanging")
            .attr("x", this.boardWidth*0.7)
            .attr("y", this.boardHeight + DataCard.unitSize)
            .text("X");
        this.X_IWM.append("tspan")
            .attr("dy", "0.5em")
            .text("IWM")
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
        const sharpness = 0.5;
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
        let duration = 100;
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
        if (this.arrowIWM === undefined) {
            this.arrowIWM = svg.append("path")
                .attr("stroke", "red")
                .attr("stroke-width", 6)
                .attr("fill", "none");
        }
        if (this.arrowRND === undefined) {
            this.arrowRND = svg.append("path")
                .attr("stroke", "red")
                .attr("stroke-width", 6)
                .attr("fill", "none");
        }

        const RNDCard = this.memoryCards[trainer.randomIdx];
        const startRND = {
            x: RNDCard.position.x + RNDCard.layout.shape.width / 2,
            y: this.gridY(this.pendingSize) + DataCard.unitSize * 1.5
        }
        const endRND = {
            x: parseInt(this.X_RND.attr("x")),
            y: parseInt(this.X_RND.attr("y"))
        }
        this.arrowRND.transition()
            .duration(150)
            .attr("d", this.createVertConnector(startRND, endRND));



        // start is the bottom of the selected MemoryCard
        const IWMIdx = d3.min([maxIndices[0], this.memorySize - 1]);
        const IWMCard = this.memoryCards[IWMIdx];
        if (IWMCard !== undefined) {
            const start = {
                x: IWMCard.position.x + IWMCard.layout.shape.width / 2,
                y: this.gridY(this.pendingSize) + DataCard.unitSize * 1.5
            }
            // end is the top of the "X_IWM"
            const end = {
                x: parseInt(this.X_IWM.attr("x")),
                y: parseInt(this.X_IWM.attr("y"))
            }
            this.arrowIWM.transition()
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
                    transitionCard.changeOrientation("vertical", 500),
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
        
        // If the capacity of the memory entries is exceeded
        // remove the oldest memory entry
        if (this.memoryCards.length > this.memorySize) {
            const oldestMemoryCard = this.memoryCards.pop();
            oldestMemoryCard.removeDOM();
        }

        await this.updateMemoryPositions();
        if (this.memoryCards.length > 0) {
            this.renderSimilarities();
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
}
const domHandler = new DOMHandler();


function frameToTensor(source) {
    // Read frame
    let frame = tf.browser.fromPixels(source)
    
    // Resize
    frame = frame.resizeNearestNeighbor([32, 32]);

    // Add batch dimension
    frame = frame.expandDims(0);

    // Normalize to [-1, 1]
    frame = frame.div(tf.scalar(128)).sub(tf.scalar(1));
    
    return frame;
}


// Start the webcam feed
async function startWebcam() {
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
}

function captureWebcam() {
    const ctx = canvas.getContext('2d');
    const outputSize = 480; // Set the desired output size

    // Set the canvas size to the output size
    canvas.width = outputSize;
    canvas.height = outputSize;

    // Calculate the coordinates to crop the video
    const videoAspectRatio = video.videoWidth / video.videoHeight;
    const outputAspectRatio = outputSize / outputSize;

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
    ctx.drawImage(video, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, outputSize, outputSize);

    const dataURL = canvas.toDataURL('image/png');
    const dataTensor = tf.tidy(() => frameToTensor(video));
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
    if (domHandler.renderPromise !== null) {
        await domHandler.renderPromise;
        
    }

    dataHandler.pendingEntries[0].label = label;
    domHandler.pendingCards[0].updateDOM();

    let inData;
    if (url !== null) {
        inData = await loadImage(url);
    }

    // outData = model.predict(inData);
    const dataEntry = tf.tidy(() => {
        if (url === null) {
            inData = captureWebcam();
        } 
        const outData = trainer.model.predict(inData.Tensor);
        const dataEntry = new DataEntry(inData, outData, label);
        dataHandler.addDataEntry(dataEntry);
        return dataEntry;
    });
    
    await domHandler.addDataCard(dataEntry);
    console.log(tf.memory().numTensors);
}


async function updatePendingDataCard(idx) {
    // Update the newest data entry
    tf.tidy(() => {
        const dataEntry = dataHandler.pendingEntries[idx];
        const inData = captureWebcam();
        const outData = trainer.model.predict(inData.Tensor);
        dataEntry.updateData(inData, outData);
        dataHandler.updateSimilaritiesRow(idx);
    });
    await domHandler.renderSimilarities();
    await domHandler.pendingCards[idx].updateDOM();
}

async function loadImages() {
    // Load all images from the folder
    // and create the datacards
    const urls = []
    for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 5; j++) {
            urls.push({
                url: `demo-pretrain-data/${i}/image${j}.png`,
                label: i
            });
        }
    }
    // swap the first and the last image
    const tmp = urls[0];
    urls[0] = urls[urls.length - 1];
    urls[urls.length - 1] = tmp;


    for (let i = 0; i < urls.length; i++) {
        const url = urls[i];
        await createDataCard(url.label, url.url);
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

        const filename = `image${categoryIndices[category]}.png`;
        // Assuming dataURL is in the format "data:image/png;base64,..."
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

let interval = null;
// Initialize the webcam on page load
document.addEventListener('DOMContentLoaded', 
    async () => {
        await Promise.all([
            trainer.initializeModel(),
            loadImages(),
            startWebcam(),
        ]);
        await createDataCard();

        // Connect the buttons
        document.getElementById("addCategory0")
            .addEventListener("click", () => createDataCard('0'));
        document.getElementById("addCategory1")
            .addEventListener("click", () => createDataCard('1'));
        document.getElementById("addCategory2")
            .addEventListener("click", () => createDataCard('2'));
        document.getElementById("trainModel")
            .addEventListener("click", () => trainer.trainModel());
        document.getElementById("saveImages")
            .addEventListener("click", downloadAllImagesAsZip);

        // Start the interval to update the pending datacard
        interval = setInterval(async () => {updatePendingDataCard(0);}, 33);
    }
);

// stop the setInterval and webcam when the user switches tabs
document.addEventListener('visibilitychange', async () => {
    if (document.hidden) {
        if (interval !== null) {
            clearInterval(interval);
            interval = null;

            // Stop the webcam
            const stream = video.srcObject;
            const tracks = stream.getTracks();
            tracks.forEach(function(track) {
                track.stop();
            });
            video.srcObject = null;

        }
    } else if (document.visibilityState === "visible") {
        if (interval === null) {
            await startWebcam();
            interval = setInterval(async () => {updatePendingDataCard(0);}, 33);
        }
    }
});


document.addEventListener('keydown', function(event) {
    let label;
    switch (event.key.toLowerCase()) {
        case 'v':
            label = UNLABELED;
            break;
        case 'b':
            label = '0';
            break;
        case 'n':
            label = '1';
            break;
        case 'm':
            label = '2';
            break;
        case 't':
            trainer.trainModel();
            return;
        default:
            return; // Do nothing if it's any other key
    }
    createDataCard(UNLABELED);
    // Handle the output as needed, such as updating the UI or triggering other actions.
});
