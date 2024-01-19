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
    constructor(inData, outData, label = null) {
        this.inData = inData; // An image or a video frame
        this.pred = outData[0]; // A [numClasses] tensor representing the class probabilities
        this.feat = outData[1]; // A [9] tensor representing the feature vector
        this.label = label; // A string or number representing the label
        this.id = DataEntry.count;
        DataEntry.count++;
    }
}

class DataCard {
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

    static maxCircleRadius = DataCard.unitSize / 3 / 2 * .9;

    constructor(dataEntry, position = {x: 0, y: 0}, orientation="horizontal") {
        this.dataEntry = dataEntry;
        this.position = position;
        this.orientation = orientation;
        this.layout = DataCard.layouts[orientation];
    }

    async createDOM() {
        const mainGroup = d3.select("#main")
            .append("g");
        
        mainGroup.attr("class", "datacard")
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

        const pos = d3.scalePoint()
            .domain(d3.range(3))
            .range([0, DataCard.unitSize])
            .padding(.5);

        const embeddings = await this.dataEntry.feat.data();
        
        featureGroup.selectAll("circle")
            .data(embeddings)
            .join("circle")
            .attr("cx", (d, i) => pos(i % 3))
            .attr("cy", (d, i) => pos(Math.floor(i / 3)))
            .attr("r", (d, i) => d * DataCard.maxCircleRadius)
            .attr("fill", "black");

        this.mainGroup = mainGroup;
        this.background = background;
        this.imageGroup = imageGroup;
        this.featureGroup = featureGroup;
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


    async changePosition(position) {
        this.position = position;
        await this.mainGroup.transition()
            .duration(500)
            .attr(
                "transform", `translate(${position.x}, ${position.y})`
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
    constructor(memorySize = 5, pendingSize = 3, width = 700, height = 500) {
        // the memory entries are the labeled datacards
        // the pending entries are the unlabeled datacards
        // the width and height are used for the inner dimensions of table

        this.memorySize = memorySize;
        this.pendingSize = pendingSize;
        this.memory_entries = [];
        this.pending_entries = [];
        this.width = width;
        this.height = height;

        this.setDOMPositions();
    }

    setDOMPositions() {
        // The following are used for the UI
        
        // the pending datacards are shown on the vertical axis on the left
        const pendingX = 0;
        this.pendingDOMPositions = [];
        const pendingY = d3.scaleBand()
            .domain(d3.range(this.pendingSize))
            .range([0, this.height - DataCard.unitSize])
            .paddingInner(0.05);
        
        // We add +1 to allow smooth moving around the corner
        for (let i = 0; i < this.pendingSize; i++) {
            const position = { x: pendingX, y: pendingY(i) };
            this.pendingDOMPositions.push(position);
        }

        this.tempPosition = { 
            x: 0, 
            y: this.height - DataCard.unitSize 
        };

        // the labeled datacards are shown on the horizontal axis on the bottom
        const memoryY = this.height - DataCard.unitSize;
        this.memoryDOMPositions = [];
        scale = d3.scaleBand()
            .domain(d3.range(this.memorySize))
            .range([DataCard.unitSize*2, this.width])
            .paddingInner(0.05);
            
        for (let i = 0; i < this.memorySize; i++) {
            const memoryX = scale(i);
            const position = { x: memoryX, y: memoryY };
            this.memoryDOMPositions.push(position);
        }

    }

    async addDataCard(dataEntry) {
        // Add card to the pending entries
        const position = this.pendingDOMPositions[0];
        const dataCard = new DataCard(dataEntry, position);
        await dataCard.createDOM();
        this.pending_entries.unshift(dataCard);
        
        let transitionCard = null
        if (this.pending_entries.length > this.pendingSize) {
            transitionCard = this.pending_entries.pop();
        }
        this.updatePendingPositions();

            
        if (transitionCard !== null) {
            // if labeled, add to the memory entries
            if (transitionCard.dataEntry.label !== null) {
                this.memory_entries.unshift(transitionCard);
                await Promise.all([
                    transitionCard.changePosition(this.tempPosition),
                    transitionCard.changeOrientation("vertical", 500)
                ]);
            } else {
                // if unlabeled, remove from the DOM
                await transitionCard.changePosition(this.tempPosition);
                transitionCard.removeDOM();
            }
        }
        
        // If the capacity of the memory entries is exceeded
        // remove the oldest memory entry
        if (this.memory_entries.length > this.memorySize) {
            const oldestMemoryCard = this.memory_entries.pop();
            oldestMemoryCard.removeDOM();
        }

        this.updateMemoryPositions();
    }

    updatePendingPositions() {
        // Update the positions of the pending entries
        for (let i = 0; i < this.pending_entries.length; i++) {
            const pendingCard = this.pending_entries[i];
            const position = this.pendingDOMPositions[i];
            pendingCard.changePosition(position);
        }
    }
    updateMemoryPositions() {
        // Update the positions of the memory entries
        for (let i = 0; i < this.memory_entries.length; i++) {
            const memoryCard = this.memory_entries[i];
            const position = this.memoryDOMPositions[i];
            memoryCard.changePosition(position);
        }
    }
}


class SimilarityMatrixHandler {
    constructor(position) {

    }
}


function initializeModel(numClasses = 3) {
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
    // console.log(feature.outputShape);
    // feature.add(tf.layers.globalAveragePooling2d({
    //     dataFormat: 'channelsLast',
    // }));
    feature.add(tf.layers.flatten());
    feature.add(tf.layers.dense({
        units: 64, 
        kernelInitializer: 'varianceScaling',
        kernelRegularizer: 'l1l2',
        activation:'selu'
    }));
    feature.add(tf.layers.dense({
        units: 9, 
        kernelInitializer: 'varianceScaling', 
        kernelRegularizer: 'l1l2',
        activation:'sigmoid'
    }));

    const classifier = tf.sequential();
    classifier.add(tf.layers.dense({
        inputShape: [9],
        units: numClasses, 
        kernelInitializer: 'varianceScaling', 
        kernelRegularizer: 'l1l2',
        activation:'softmax'
    }));
    
    feat = feature.apply(input);
    pred = classifier.apply(feat);
    
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
function startWebcam() {
    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
            })
            .catch(error => {
                console.error('Error accessing the webcam', error);
            });
    }

    // Add event listener to the webcam feed
    video.addEventListener('loadeddata', () => {
        console.log('Video loaded');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
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


// Initialize the webcam on page load
document.addEventListener('DOMContentLoaded', 
    () => {
        startWebcam();
        model = initializeModel();
    }
);

async function createDataCard() {
    const inData = captureWebcam();
    const outData = model.predict(inData.Tensor);
    const dataEntry = new DataEntry(inData, outData);
    DOMHandler.addDataCard(dataEntry);
}


// When the user clicks on the "Capture" button create a data entry
const DOMHandler = new DOMHandler();
document.getElementById('capture').addEventListener('click', async () => {
    await createDataCard();
});

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