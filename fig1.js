
function fig1() {

let defs = d3.select("#defs");
let svg = d3.select("#fig1-svg").attr("stroke", "black")
const width = svg.attr("viewBox").split(" ")[2];
const height = svg.attr("viewBox").split(" ")[3];
let baseY = 100;
let labelGap = 100;
let keywords = [
    "beaver", "dolphin", "otter", "seal", "whale",
    "aquarium fish", "flatfish", "ray", "shark", "trout",
    "orchids", "poppies", "roses", "sunflowers", "tulips",
    "bottles", "bowls", "cans", "cups", "plates",
    "apples", "mushrooms", "oranges", "pears", "peppers",
    "clock", "lamp", "telephone", "TV",
    "bed", "chair", "couch", "table", "wardrobe",
    "bee", "beetle", "butterfly", "caterpillar", "cockroach",
    "bear", "leopard", "lion", "tiger", "wolf",
    "bridge", "castle", "house", "road", "skyscraper",
    "cloud", "forest", "mountain", "plain", "sea",
    "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
    "fox", "porcupine", "possum", "raccoon", "skunk",
    "crab", "lobster", "snail", "spider", "worm",
    "baby", "boy", "girl", "man", "woman",
    "crocodile", "dinosaur", "lizard", "snake", "turtle",
    "hamster", "mouse", "rabbit", "shrew", "squirrel",
    "maple", "oak", "palm", "pine", "willow",
    "bicycle", "bus", "motorcycle", "pickup truck", "train",
    "lawn-mower", "rocket", "streetcar", "tank", "tractor"
]

function getRandomImage(randomKeyword) {
    return `https://source.unsplash.com/random?${randomKeyword}`;
}

function makeCurlyBrace(x1,y1,x2,y2,w,q){
    //Calculate unit vector
    var dx = x1-x2;
    var dy = y1-y2;
    var len = Math.sqrt(dx*dx + dy*dy);
    dx = dx / len;
    dy = dy / len;

    //Calculate Control Points of path,
    var qx1 = x1 + q*w*dy;
    var qy1 = y1 - q*w*dx;
    var qx2 = (x1 - .25*len*dx) + (1-q)*w*dy;
    var qy2 = (y1 - .25*len*dy) - (1-q)*w*dx;
    var tx1 = (x1 -  .5*len*dx) + w*dy;
    var ty1 = (y1 -  .5*len*dy) - w*dx;
    var qx3 = x2 + q*w*dy;
    var qy3 = y2 - q*w*dx;
    var qx4 = (x1 - .75*len*dx) + (1-q)*w*dy;
    var qy4 = (y1 - .75*len*dy) - (1-q)*w*dx;

    return ( "M " +  x1 + " " +  y1 +
        " Q " + qx1 + " " + qy1 + " " + qx2 + " " + qy2 + 
        " T " + tx1 + " " + ty1 +
        " M " +  x2 + " " +  y2 +
        " Q " + qx3 + " " + qy3 + " " + qx4 + " " + qy4 + 
        " T " + tx1 + " " + ty1 );
}

function getImageSize(imageUrl) {
    return new Promise((resolve, reject) => {
        const img = new Image();

        img.onload = function() {
            const size = {
                width: img.width,
                height: img.height
            };
            resolve(size);
        };

        img.onerror = function() {
            reject(new Error('Failed to load image.'));
        };

        img.src = imageUrl;
    });
}

const cloudImage = svg.append("image")
    .attr('xlink:href', 'cloud.png')
    .attr('x', width - 200)
    .attr('y', baseY - 50)
    .attr('width', 200)
    .attr('height', 200);


// Define background rectangle for the image
const annotImageBG = svg.append("rect")
    .attr('x', width - 200)
    .attr('y', baseY + labelGap + 50)
    .attr('width', 200)
    .attr('height', 180)
    .attr('fill', 'white')
    .attr('stroke', 'white');


const annotImage = svg.append("image")
    .attr('xlink:href', 'annot.png')
    .attr('x', width - 220)
    .attr('y', baseY + labelGap + 50)
    .attr('width', 200)
    .attr('height', 200);



const rectSize = 100;
const timelineHeight = height - 150;
const spacing = 10;
const shiftAmount = rectSize + spacing;
const shiftDuration = 1000;

let labelDelay = 4;

let rectID = 0;
const matrix = d3.select("#matrix");

function addRectangle() {
    let randomKeyword = keywords[Math.floor(Math.random() * keywords.length)];
    let imageUrl = getRandomImage(randomKeyword);

    svg.selectAll(".sample-group")
        .transition()
        .duration(shiftDuration)
        .attr("transform", function() {
            g = d3.select(this);
            const currentX = parseFloat(d3.select(this).attr("data-x")) - shiftAmount;
            g.attr("data-x", currentX);

            const currentStrides = parseInt(d3.select(this).attr("data-strides"), 10) + 1;
            g.attr("data-strides", currentStrides);
            const progressWidth = (Math.min(currentStrides-1, labelDelay) / labelDelay) * rectSize;
            g.select(".progress-bar").transition().attr("width", progressWidth);

            if (currentStrides >= labelDelay + 2) {
                g.select(".image-sample").classed("grayscale", false);
                g.select(".label-group").transition().duration(1000)
                    .attr("transform", `translate(0, ${-labelGap + 10})`);
                g.select(".label-text").transition().duration(1000).style("opacity", 1);
                g.select("rect").transition().duration(1000).style("fill", "#A7D397");
            }

            return `translate(${currentX}, 0)`;
        })
        .on("end", function() {
            if (parseFloat(d3.select(this).attr("data-x")) + rectSize * 4 < 0) {
                d3.select(this).remove();
            }
        });

    // Define the clipping path
    const clip = svg.append("defs")
        .append("clipPath")
        .attr("id", "clip-rounded-rect")
        .append("rect")
        .attr("x", 0)
        .attr("y", baseY)
        .attr("width", rectSize)
        .attr("height", rectSize)
        .attr("rx", 15)
        .attr("ry", 15);

    // Create a group
    const group = svg.append("g")
        .attr("class", "sample-group")
        .attr("data-x", width - 200)
        .attr("data-strides", "0")
        .attr("transform", `translate(${width}, 0)`);

    // Add the rounded rectangle
    group.append("rect")
        .attr("x", 0)
        .attr("y", baseY)
        .attr("width", rectSize)
        .attr("height", rectSize)
        .attr("rx", 15)
        .attr("ry", 15)
        .style("stroke", "black")
        .style("stroke-width", "2px")
        .style("fill", "lightgray");


    imageSample = group.append("image")
        .attr("class", "image-sample")
        .attr("xlink:href", imageUrl)
        .attr("x", 0)  // Adjust starting x coordinate based on crop details
        .attr("y", baseY)  // Adjust starting y coordinate based on crop details
        .attr("clip-path", "url(#clip-rounded-rect)")
        .classed("grayscale", true);
    
    // Determine the aspect ratio of the image
    getImageSize(imageUrl).then(size => {
        const aspectRatio = size.width / size.height;
        if (aspectRatio > 1) {
            imageSample
                .attr("height", rectSize)
                .attr("x", (rectSize - rectSize * aspectRatio) / 2);
        } else {
            imageSample
                .attr("width", rectSize)
                .attr("y", baseY + (rectSize - rectSize / aspectRatio) / 2);
        }

    });

    labelgroup = group.append("g")
        .attr("class", "label-group")
        .attr("transform", "translate(0, 0)");
        
        // Container rectangle
        labelgroup.append("rect")
        .attr("class", "label-container")
        .attr("x", 0)
        .attr("y", baseY + rectSize + labelGap)
        .attr("width", rectSize)
        .attr("height", 20)
        .attr("rx", 10)
        .attr("ry", 10)
        .style("fill", "white")
        .style("stroke", "black")
        .style("stroke-width", "2px");

        // Progress bar rectangle
        labelgroup.append("rect")
            .attr("class", "progress-bar")
            .attr("x", 0)
            .attr("y", baseY + rectSize + labelGap)
            .attr("width", 10)
            .attr("height", 20)
            .attr("rx", 10)
            .attr("ry", 10)
            .text(imageUrl)
            .attr("stroke-width", "2px")
            .style("fill", "#A7D397");
    
        labelText = labelgroup.append("text")
            .attr("class", "label-text")
            .attr("x", rectSize / 2)
            .attr("y", baseY + rectSize + labelGap + 11)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .style("font-size", "17px")
            .text(randomKeyword)
            .style("opacity", 0)
            .style("font-weight", "light");


    // Add the tick on the timeline
    tick = group.append("g")
        .attr("class", "timeline-tick");
    
    
    tick.append("line")
        .attr("x1", rectSize / 2)
        .attr("y1", timelineHeight - 5)
        .attr("x2", rectSize / 2)
        .attr("y2", timelineHeight + 5)
        .attr("stroke", "black")
        .attr("stroke-width", 2);

    ticktext = tick.append("text")
        .attr("x", rectSize / 2)
        .attr("y", timelineHeight + 20)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .style("font-size", "25px")
        .text(rectID+1);

    if (rectID+1 === 69) {
        // Noice
        ticktext.style("font-weight", "bold").style("font-size", "30px");

    }
    
    
    cloudImage.raise();
    annotImageBG.raise();
    annotImage.raise();
    rectID++;
}

let interval = setInterval(addRectangle, shiftDuration);
// stop the setInterval and webcam when the user switches tabs
document.addEventListener('visibilitychange', async () => {
    if (document.hidden) {
        if (interval !== null) {
            clearInterval(interval);
            interval = null;
        }
    } else if (document.visibilityState === "visible") {
        if (interval === null) {
            interval = setInterval(
                addRectangle, 
                shiftDuration
            );
        }
    }
});



timeline = svg.append("g")
    .attr("class", "timeline")
    .attr("transform", "translate(0, 0)");

timeline.append("line")
    .attr("x1",0)  
    .attr("y1",timelineHeight)
    .attr("x2",width-220)
    .attr("y2",timelineHeight)  
    .attr("stroke","black")  
    .attr("stroke-width",2)  
    .attr("marker-end","url(#arrow)");  

timeline.append("text")
    .attr("x", 70)
    .attr("y", timelineHeight + 60)
    .attr("text-anchor", "start")
    .attr("dominant-baseline", "middle")
    .style("font-size", "30px")
    .text("Time step");

delayed_legend = timeline.append("g")
    .attr("class", "delayed-legend")
    .attr("transform", "translate(0, 0)");

// Draw curly brace for the label delay
// brace_x1 = 470;
// brace_x2 = 780;
let brace_x1 = width - annotImage.attr("width") - labelDelay * shiftAmount;
let brace_x2 = width - annotImage.attr("width") - spacing;
brace_y = timelineHeight + 50;
brace_xmid = (brace_x1 + brace_x2) / 2;
const brace = delayed_legend.append("path")
    .attr("d", makeCurlyBrace(brace_x1, brace_y, brace_x2, brace_y, 25, 0.6))
    .attr("stroke", "black")
    .attr("stroke-width", 2)
    .attr("fill", "none")

const brace_text = delayed_legend.append("text")
    .attr("x", brace_xmid)
    .attr("y", timelineHeight + 100)
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "middle")
    .style("font-size", "20px")
    // make the label delay text dependent on the label delay
    // .text(function() {
    //     return `Label delay (d=${labelDelay})`;
    // });
    .text(`Label delay (d=${labelDelay})`);


// Draw dashed line for train / eval
delayed_legend.append("line")
    .attr("x1", width - 315)
    .attr("y1", baseY - 70)
    .attr("x2", width - 315)
    .attr("y2", timelineHeight)
    .attr("stroke", "black")
    .attr("stroke-width", 2)
    .attr("stroke-dasharray", "5,5");

delayed_legend.append("text")
    .attr("x", width - 210 - rectSize / 2)
    .attr("y", baseY - 50)
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "middle")
    .style("font-size", "30px")
    .text("Eval");

delayed_legend.append("text")
    .attr("x", width - 220 - rectSize * 1.5)
    .attr("y", baseY - 50)
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "middle")
    .style("font-size", "30px")
    .text("Train");

// add a button to change the label delay
function increaseLabelDelay() {
    labelDelay = labelDelay % 5 + 1;
    
    
    // update the curly brace
    old_brace_x1 = brace_x1;
    brace_x1 = width - annotImage.attr("width") - labelDelay * shiftAmount;
    brace_x2 = width - annotImage.attr("width") - spacing;
    brace_y = timelineHeight + 50;
    brace_xmid = (brace_x1 + brace_x2) / 2;

    brace_text.text(function() {
        return `Label delay (d=${labelDelay})`;
    });

    // make smooth transition
    brace.transition()
        .duration(500)
        .attrTween("d", function() {
            return function(t) {
                console.log(t);
                console.log(old_brace_x1);
                console.log(brace_x1);
                curr_brace_x1 = old_brace_x1 * (1 - t) + brace_x1 * t;
                console.log(curr_brace_x1);
                return makeCurlyBrace(curr_brace_x1, brace_y, brace_x2, brace_y, 25, 0.6);
            }
        });

    brace_text.transition()
        .duration(500)
        .attr("x", brace_xmid);

}

 
    delayed_legend.append("rect")
        .attr("class", "button-rect")
        .attr("x", 25)
        .attr("y", 25)
        .attr("width", rectSize * 2)
        .attr("height", 50)
        .attr("rx", 10)
        .attr("ry", 10)
        .on("click", increaseLabelDelay);

    delayed_legend.append("text")
        .attr("class", "button-text")
        .attr("x", 25 + rectSize)
        .attr("y", 25 + 25)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .style("font-size", "20px")
        .text("Change label delay");



    // make all delayed legend elements invisible
    delayed_legend.selectAll("*")
        .style("opacity", 0)
        .transition()
        .duration(5000)
        .style("opacity", 0)
        .transition()
        .duration(5000)
        .style("opacity", 1);
}

fig1();