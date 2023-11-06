function fig6() {
    const svg = d3.select("#fig6-svg").attr("stroke", "black")
    const baseY = 180;
    const rectSize = 50;
    const labelGap = 100;
    const labelY = baseY + rectSize + labelGap;
    const timelineY = (labelY + baseY) / 2 + rectSize/2;
    const spacing = 10;
    const shiftAmount = rectSize + spacing;

    let baseShiftDuration = 1000;
    let shiftDurationSpeedup = 50;
    let labelDelay = 4;
    let streamRate = 5;
    let numWorkers = 2;
    let shiftDuration = baseShiftDuration - shiftDurationSpeedup * streamRate;
    
    let rectID = 0;
    const streamFigSize = 110;

    
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
    
    
    // Define background rectangle for the image
    const annotImageBG = svg.append("rect")
        .attr('x', svg.attr("width") - streamFigSize)
        .attr('y', 0)
        .attr('width', streamFigSize)
        .attr('height', labelY + 20)
        .attr('fill', 'white')
        .attr('stroke', 'white');

    const cloudImage = svg.append("image")
        .attr('xlink:href', 'bare-cloud.png')
        .attr('x', svg.attr("width") - streamFigSize)
        .attr('y', baseY - 20)
        .attr('width', streamFigSize)
        .attr('height', streamFigSize);



    const annotImage = svg.append("image")
        .attr('xlink:href', 'doctor.png')
        .attr('x', svg.attr("width") - streamFigSize)
        .attr('y', baseY + labelGap)
        .attr('width', streamFigSize)
        .attr('height', streamFigSize);

    const rateText = svg.append("text")
        .attr("x", svg.attr("width") - streamFigSize / 2)
        .attr("y", baseY + streamFigSize - 10)
        .attr("text-anchor", "middle")
        // .attr("dominant-baseline", "middle")
        .style("font-size", "20px")
        .text(`r = ${streamRate}:${numWorkers}`);






    function addRectangle() {

        svg.selectAll(".sample-group")
            .transition()
            .duration(shiftDuration)
            .attr("transform", function() {
                g = d3.select(this);
                const currentX = parseFloat(g.attr("data-x")) - shiftAmount;
                g.attr("data-x", currentX);

                const currentStrides = parseInt(g.attr("data-strides"), 10) + 1;
                // isAnnotated = parseInt(g.attr("sampleID"), 10) % streamRate == 0;
                // just read from attr
                isAnnotated = g.attr("isAnnotated") == "true";
                g.attr("data-strides", currentStrides);
                if (isAnnotated) {
                    const progressWidth = (Math.min(currentStrides-1, labelDelay) / labelDelay) * rectSize;
                    g.select(".progress-bar").transition().attr("width", progressWidth);
                    transitionTime = Math.min(1000, shiftDuration);
                    if (currentStrides >= labelDelay + 2) {
                        g.select(".image-sample").classed("grayscale", false);
                        g.select(".label-group").transition().duration(transitionTime)
                            .attr("transform", `translate(0, ${-labelGap + 10})`);
                        g.select(".label-text").transition().duration(transitionTime).style("opacity", 0);
                        g.select("rect").transition().duration(transitionTime).style("fill", "#ffb55a");
                    }
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
        let workerID = rectID % streamRate;
        isAnnotated = workerID <= numWorkers - 1;
        const group = svg.append("g")
            .attr("class", "sample-group")
            .attr("sampleID", rectID)
            .attr("isAnnotated", isAnnotated)
            .attr("data-x", svg.attr("width") - streamFigSize)
            .attr("data-strides", "0")
            .attr("transform", `translate(${svg.attr("width")}, 0)`);

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


        if (isAnnotated) {
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
                    .attr("stroke-width", "2px")
                    .style("fill", "#ffb55a");
                
                labelText = labelgroup.append("text")
                    .attr("class", "label-text")
                    .attr("x", rectSize / 2)
                    .attr("y", baseY + rectSize + labelGap + 10)
                    .attr("text-anchor", "middle")
                    .attr("dominant-baseline", "middle")
                    .style("font-size", "18px")
                    .text(`#${workerID+1}`);
        }


        // Add the tick on the timeline
        tick = group.append("g")
            .attr("class", "timeline-tick");
        
        
        tick.append("line")
            .attr("x1", rectSize / 2)
            .attr("y1", timelineY - 5)
            .attr("x2", rectSize / 2)
            .attr("y2", timelineY + 5)
            .attr("stroke", "black")
            .attr("stroke-width", 2);

        ticktext = tick.append("text")
            .attr("x", rectSize / 2)
            .attr("y", timelineY + 20)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .style("font-size", "15px")
            .text(rectID+1);

        if (rectID+1 === 69) {
            // Noice
            ticktext.style("font-weight", "bold").style("font-size", "30px");

        }
        
        
        annotImageBG.raise();
        cloudImage.raise();
        annotImage.raise();
        rateText.raise();
        rectID++;
    }

    let intervalId = setInterval(addRectangle, shiftDuration);


    timeline = svg.append("g")
        .attr("class", "timeline")
        .attr("transform", "translate(0, 0)");

    timeline.append("line")
        .attr("x1",0)  
        .attr("y1",timelineY)
        .attr("x2",svg.attr("width")-streamFigSize-2*spacing)
        .attr("y2",timelineY)  
        .attr("stroke","black")  
        .attr("stroke-width",2)  
        .attr("marker-end","url(#arrow)");  

    delayed_legend = timeline.append("g")
        .attr("class", "delayed-legend")
        .attr("transform", "translate(0, 0)");

    brace_x1 = svg.attr("width") - annotImage.attr("width") - labelDelay * shiftAmount;
    brace_x2 = svg.attr("width") - annotImage.attr("width") - spacing;
    brace_y = labelY+30;
    brace_xmid = (brace_x1 + brace_x2) / 2;
    brace = delayed_legend.append("path")
        .attr("d", makeCurlyBrace(brace_x1, brace_y, brace_x2, brace_y, 25, 0.6))
        .attr("stroke", "black")
        .attr("stroke-width", 2)
        .attr("fill", "none")

    brace_text = delayed_legend.append("text")
        .attr("x", brace_xmid)
        .attr("y", labelY + 70)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .style("font-size", "20px")
        // make the label delay text dependent on the label delay
        // .text(function() {
        //     return `Label delay (d=${labelDelay})`;
        // });
        .text(`d=${labelDelay}`);
    // Show two sliders, one for the label delay
    // and one for the rate of the annotation stream (data:annot which is always N:1)
    // if more data pours in than we can annotate, we will have to leave some data unannotated

    function changeLabelDelay(newLabelDelay) {
        labelDelay = newLabelDelay;
        // update the curly brace
        old_brace_x1 = brace_x1;
        brace_x1 = svg.attr("width") - annotImage.attr("width") - labelDelay * shiftAmount;
        brace_x2 = svg.attr("width") - annotImage.attr("width") - spacing;
        brace_y = labelY + 30;
        brace_xmid = (brace_x1 + brace_x2) / 2;

        brace_text.text(function() {
            return `d=${labelDelay}`;
        });

        // make smooth transition
        brace.transition()
            .duration(500)
            .attrTween("d", function() {
                return function(t) {
                    curr_brace_x1 = old_brace_x1 * (1 - t) + brace_x1 * t;
                    console.log(curr_brace_x1);
                    return makeCurlyBrace(curr_brace_x1, brace_y, brace_x2, brace_y, 25, 0.6);
                }
            });

        brace_text.transition()
            .duration(500)
            .attr("x", brace_xmid);

    }

    function changeStreamRate(newStreamRate) {
        streamRate = newStreamRate;
        shiftDuration = baseShiftDuration - shiftDurationSpeedup * streamRate;
        clearInterval(intervalId);
        intervalId = setInterval(addRectangle, shiftDuration);
        rateText.text(`r = ${streamRate}:${numWorkers}`)
            .transition()
            .duration(500);
    }


    function changeAnnotRate(newAnnotRate) {
        numWorkers = newAnnotRate;
        rateText.text(`r = ${streamRate}:${numWorkers}`)
            .transition()
            .duration(500);
    }

    function createSlider(title, startX, startY, width, values, defaultValue, callback) {
        // const width = 400;
        // const startX = 50;
        // const startY = 50;

        // Discrete values
        // const values = [1, 2, 3, 4];

        // Create a scale for the slider with discrete steps
        const scale = d3.scaleLinear()
            .domain([values[0], values[values.length - 1]])
            .range([0, width])
            .clamp(true);


        // Create a group for the slider
        const slider = svg.append('g')
            .attr('class', 'slider')
            .attr('transform', `translate(${startX}, ${startY})`);


        // Add the text label for the slider
        slider.append('text')
            .attr('class', 'label')
            .attr('x', 0)
            .attr('y', -10)
            .attr('text-anchor', 'left')
            .text(title);

        // Add the track as a single straight thin black line
        slider.append('line')
            .attr('class', 'track')
            .attr('x1', 0)
            .attr('x2', width)
            .attr('y1', 5)
            .attr('y2', 5)
            .style('stroke', '#000')
            .style('stroke-width', '2px');

        // Add short black vertical ticks for each discrete value
        values.forEach(value => {
            slider.append('line')
                .attr('x1', scale(value))
                .attr('x2', scale(value))
                .attr('y1', 0)
                .attr('y2', 10)
                .style('stroke', '#000')
                .style('stroke-width', '2px');
            
            // Add the text for each value
            slider.append('text')
                .attr('x', scale(value))
                .attr('y', 30)
                .attr('text-anchor', 'middle')
                .text(value);
        });

        slider.append('rect')
            .attr('class', 'overlay')
            .attr('x', scale.range()[0])
            .attr('width', scale.range()[1] - scale.range()[0])
            .attr('y', 0) // Match the tick height
            .attr('height', 30)
            .style('fill', 'transparent')
            .style('cursor', 'pointer')
            .style('stroke', 'transparent')
            .style('pointer-events', 'all')
            .on('click', function() {
                // On click, update the position of the handle
                const x = d3.mouse(this)[0];
                const value = Math.round(scale.invert(x));
                handle.transition().duration(100).attr('cx', scale(value));
                callback(value);
            });

        // Add the handle as a circle
        const handle = slider.append('circle')
            .attr('class', 'handle')
            .attr('cx', scale(defaultValue))
            .attr('cy', 5)
            .attr('r', 6)
            .style('fill', '#fff')
            .style('cursor', 'pointer')
            .style('stroke', '#000')
            .style('stroke-width', '3px');

        // Drag behavior for D3.js version 5
        const drag = d3.drag()
            .on('drag', function() {
                // Get the nearest value to the current x position of the drag
                const x = d3.event.x;
                const value = Math.round(scale.invert(x));
                handle.transition().duration(50).attr('cx', scale(value));
                callback(value);
            });

        handle.call(drag);
    }
    // createSlider("Label delay (d)", 50, 40, 400, [1, 5], labelDelay, changeLabelDelay);

    createSlider("Data collection rate", 50, 40, 500, [1, 2, 3, 4, 5], streamRate, changeStreamRate);
    createSlider("Annotation rate", 50, 120, 250, [1, 2, 3], numWorkers, changeAnnotRate);


}

fig6();