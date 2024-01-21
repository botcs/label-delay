const PENDING_SIZE = 3;
const MEMORY_SIZE = 8;
const NUM_CLASSES = 3;
const NUM_FEATURES = 3**2;
const UNLABELED = 42;

const video = document.getElementById('webcam');
const canvas = document.getElementById('aux-canvas');
const svg = d3.select("#main")

// Define the clipping path
const clip = svg.append("defs")
    .append("clipPath")
    .attr("id", "clip-rounded-rect")
    .append("rect")


class DataEntry {
    static count = 0;
    constructor(inData, outData, label = UNLABELED) {
        this.inData = inData; // An image or a video frame

        this.inData.Tensor = tf.variable(inData.Tensor);
        // A [NUM_CLASES] tensor representing the class probabilities
        this.pred = tf.variable(tf.squeeze(outData[0]));

        // A [NUM_FEATURES] tensor representing the feature vector
        feat = outData[1];
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
        feat = outData[1];
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
}
const dataHandler = new DataHandler();


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
        width = 1000, 
        height = 500
    ) {
        // the memory entries are the labeled datacards
        // the pending entries are the unlabeled datacards
        // the width and height are used for the inner dimensions of table

        this.memorySize = memorySize;
        this.pendingSize = pendingSize;
        this.memoryCards = [];
        this.pendingCards = [];
        this.offset = offset;
        this.width = width;
        this.height = height;

        this.setDOMPositions();
        this.mainGroup = svg.append("g")
            .attr("id", "DOMHandler")
            .attr("transform", `translate(${offset.x}, ${offset.y})`);
        this.mainGroup.append("rect")
            .attr("fill", "lightgray")
            .attr("width", width)
            .attr("height", height);

        this.similarityGroup = this.mainGroup.append("g")
            .attr("id", "similarityGroup");

        this.renderPromise = null;
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
        const endX = this.width - DataCard.unitSize*0.5;
        const endY = this.height - DataCard.unitSize*1.5;


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
    async renderSimilarities() {
        if (this.memoryCards.length === 0) {
            return;
        }
        const scores = await dataHandler.scores.array();
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
            return .8;
        }
    
        this.similarityGroup.selectAll("rect")
            .data(scores.flat())
            .join("rect")
            .transition()
            .duration(100)
            .attr("x", (d, i) => barX(i))
            .attr("y", (d, i) => barY(d, i)) // Updated y attribute
            .attr("width", barWidth)
            .attr("height", (d, i) => barHeightScale(d, i)) // Updated height attribute
            .attr("fill", (d, i) => "#1f77b4")
            .style("opacity", (d, i) => barOpacity(d, i))
            .style("stroke", "#13496f")
            .style("stroke-width", "2px")
            .attr("rx", 10)
            .attr("ry", 10);
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



function initializeModel(numClasses = NUM_CLASSES) {
    const input = tf.input({
        shape: [32, 32, 3],
        dataFormat: 'channelsLast',
    });
    const feature = tf.sequential();
    feature.add(tf.layers.conv2d({
        inputShape: [32, 32, 3],
        kernelSize: 5,
        filters: 128,
        strides: 1,
        activation: 'selu',
        kernelInitializer: 'varianceScaling',
        kernelRegularizer: 'l1l2',
        padding: 'same',
    }));
    feature.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    feature.add(tf.layers.conv2d({
        kernelSize: 3,
        filters: 128,
        strides: 1,
        activation: 'selu',
        kernelInitializer: 'varianceScaling',
        kernelRegularizer: 'l1l2',
        padding: 'same',
    }));
    feature.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    feature.add(tf.layers.conv2d({
        kernelSize: 3,
        filters: 1,
        strides: 1,
        activation: 'selu',
        kernelInitializer: 'varianceScaling',
        kernelRegularizer: 'l1l2',
        padding: 'same',
    }));
    feature.add(tf.layers.flatten());
    feature.add(tf.layers.dense({
        units: 64, 
        kernelInitializer: 'varianceScaling',
        kernelRegularizer: 'l1l2',
        activation:'selu'
    }));
    feature.add(tf.layers.dense({
        units: NUM_FEATURES, 
        kernelInitializer: 'varianceScaling', 
        kernelRegularizer: 'l1l2',
        activation:'tanh'
    }));

    const logit = tf.sequential();
    logit.add(tf.layers.dense({
        inputShape: [NUM_FEATURES],
        units: numClasses, 
        kernelInitializer: 'varianceScaling', 
        kernelRegularizer: 'l1l2',
    }));
    
    feat = feature.apply(input);
    pred = logit.apply(feat);

    
    
    const model = tf.model({
        inputs: input, 
        // outputs: {
        //     pred: pred,
        //     feat: feat
        // }
        outputs: [pred, feat]
    });
    
    const noopLoss = (yTrue, yPred) => tf.zeros([1]);
    model.compile({
        optimizer: 'adam',
        // loss: {
        //     pred: 'categoricalCrossentropy',
        //     feat: noopLoss
        // },
        loss: ['categoricalCrossentropy', noopLoss],
        metrics: ['accuracy']
    });

    // Warm up the model
    const warmupData = tf.zeros([1, 32, 32, 3]);
    model.predict(warmupData);
    warmupData.dispose();

    // Summary
    console.log(model.summary());

    return model;

}

function frameToTensor(video) {
    // Read frame
    let frame = tf.browser.fromPixels(video)
    
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
    const dataTensor = frameToTensor(video); // Assuming this is part of your existing code
    const frame = {
        dataURL: dataURL,
        Tensor: dataTensor
    };
    return frame;
}

// inData = tf.zeros([1, 32, 32, 3]);
async function createDataCard(label = UNLABELED) {
    if (domHandler.renderPromise !== null) {
        return;
    }

    // outData = model.predict(inData);
    const dataEntry = tf.tidy(() => {
        const inData = captureWebcam();
        const outData = model.predict(inData.Tensor);
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
        const outData = model.predict(inData.Tensor);
        dataEntry.updateData(inData, outData);
        dataHandler.updateSimilaritiesRow(idx);
    });
    await domHandler.renderSimilarities();
    await domHandler.pendingCards[idx].updateDOM();
}


async function trainModel() {
    // Train the model
    const optimizer = model.optimizer;
    const lossFunction = tf.losses.softmaxCrossEntropy;
    const batchSize = 2;

    const indices = [];
    for (let i = 0; i < batchSize; i++) {
        const idx = Math.floor(Math.random() * dataHandler.memoryEntries.length);
        indices.push(idx);
    }
    const data = tf.tidy(() => {
        const data = [];
        for (let i = 0; i < indices.length; i++) {
            const dataEntry = dataHandler.memoryEntries[indices[i]];
            data.push(dataEntry.inData.Tensor);
        }
        return tf.concat(data, 0);
    });
    const labels = tf.tidy(() => {
        const labels = [];
        for (let i = 0; i < indices.length; i++) {
            const dataEntry = dataHandler.memoryEntries[indices[i]];
            labels.push(parseInt(dataEntry.label));
        }
        return tf.oneHot(labels, NUM_CLASSES);
    });
    const loss = optimizer.minimize(() => {
        const logits = model.predict(data)[0];
        const loss = lossFunction(labels, logits);
        loss.print();
        return loss;
    });
    console.log(loss);
}

// Initialize the webcam on page load
document.addEventListener('DOMContentLoaded', 
    async () => {
        await startWebcam();
        model = initializeModel();
        await createDataCard();
        setInterval(async () => {updatePendingDataCard(0);}, 33);
    }
);

// Periodically update the last data entry


// // When the user clicks on the "Capture" button create a data entry
// dataEntries = [];
// document.getElementById('capture').addEventListener('click', () => {
//     const dataEntry = new DataEntry(captureWebcam());
//     dataEntries.push(dataEntry);
//     console.log(dataEntries);
//     d3.select('#data-cards')
//         .selectAll('.datacard')
//         .data(dataEntries)
//         .join('div')
//         .attr('class', 'datacard')
//         .html(d => `<img src="${d.inData.dataURL}" />`);
// });
document.addEventListener('keydown', async function(event) {
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
        default:
            return; // Do nothing if it's any other key
    }
    await createDataCard(label);
    
    // Handle the output as needed, such as updating the UI or triggering other actions.
});
