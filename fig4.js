function fig4() {
    const baseY = 10;
    const rectSize = 50;
    const labelGap = 100;
    const labelY = baseY + rectSize + labelGap;
    const timelineY = (labelY - baseY) / 2 + 50;
    const spacing = 10;
    const shiftAmount = rectSize + spacing;
    const shiftDuration = 1100;
    
    let labelDelay = 2;
    
    let rectID = 0;
    const matrix = d3.select("#matrix");
    const streamFigSize = 110;

    const defs = d3.select("#fig4-defs");
    const svg = d3.select("#fig4-svg").attr("stroke", "black")

    const width = svg.attr("viewBox").split(" ")[2];
    const height = svg.attr("viewBox").split(" ")[3];
    
    let keywords = ["Fair", "Copy", "Suit", "Infr", "Ties", "Pact", "User"];


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
    
    
    // Define background rectangle for the image
    const annotImageBG = svg.append("rect")
        .attr('x', width - streamFigSize)
        .attr('y', 0)
        .attr('width', streamFigSize)
        .attr('height', labelY + 20)
        .attr('fill', 'white')
        .attr('stroke', 'white');

    const cloudImage = svg.append("image")
        .attr('xlink:href', 'bare-cloud.png')
        .attr('x', width - streamFigSize)
        .attr('y', baseY - 20)
        .attr('width', streamFigSize)
        .attr('height', streamFigSize);



    const annotImage = svg.append("image")
        .attr('xlink:href', 'jury.png')
        .attr('x', width - streamFigSize)
        .attr('y', baseY + labelGap)
        .attr('width', streamFigSize)
        .attr('height', streamFigSize);






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
                    g.select("rect").transition().duration(1000).style("fill", "#bd7ebe");
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
            .attr("data-x", width - streamFigSize)
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
                .style("fill", "#bd7ebe");
        
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
        rectID++;
    }

    let interval = setInterval(addRectangle, shiftDuration);

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
        .attr("y1",timelineY)
        .attr("x2",width-streamFigSize-2*spacing)
        .attr("y2",timelineY)  
        .attr("stroke","black")  
        .attr("stroke-width",2)  
        .attr("marker-end","url(#arrow)");  

    delayed_legend = timeline.append("g")
        .attr("class", "delayed-legend")
        .attr("transform", "translate(0, 0)");

    // Draw curly brace for the label delay
    // brace_x1 = 470;
    // brace_x2 = 780;
    brace_x1 = width - annotImage.attr("width") - labelDelay * shiftAmount;
    brace_x2 = width - annotImage.attr("width") - spacing;
    brace_y = labelY + 50;
    brace_xmid = (brace_x1 + brace_x2) / 2;
    brace = delayed_legend.append("path")
        .attr("d", makeCurlyBrace(brace_x1, brace_y, brace_x2, brace_y, 25, 0.6))
        .attr("stroke", "black")
        .attr("stroke-width", 2)
        .attr("fill", "none")

    brace_text = delayed_legend.append("text")
        .attr("x", brace_xmid)
        .attr("y", labelY + 100)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .style("font-size", "20px")
        // make the label delay text dependent on the label delay
        // .text(function() {
        //     return `Label delay (d=${labelDelay})`;
        // });
        .text(`Label delay (d=${labelDelay})`);

    
    svg.append("text")
        .attr("x", 10)
        .attr("y", height - 30)
        .attr("text-anchor", "start")
        .attr("dominant-baseline", "middle")
        .style("font-size", "45px")
        .text("Copyright claims");

}

fig4();