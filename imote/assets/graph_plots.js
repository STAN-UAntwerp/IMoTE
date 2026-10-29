// Node plots pinned on top of the cytoscape graph.
//
// The plots are not cytoscape elements: they live in an overlay div on top of the graph canvas, so they never take
// part in the dagre layout. The overlay follows the pan, zoom and node positions of the graph. Each plot is connected
// to its node with a square bracket around the node and two lines to the near corners of the plot (a "magnifier").
//
// Dash calls window.imoteGraphPlots.render with the plots in the store (see b_node_callbacks.py), each plot is
// {id: cytoscape node id, src: image url, width: width in graph units, aspect: width / height}.
(function () {
    const SVG_NS = "http://www.w3.org/2000/svg";
    const GAP = 60;          // graph units between a node and its new plot
    const BRACKET_PAD = 6;   // graph units between the node and its bracket
    const BRACKET_ARM = 7;   // graph units, length of the bracket ends
    const DRAG_THRESHOLD = 3;  // screen px, a shorter drag is a click that selects the node
    const MAX_RETRIES = 50;

    let cy = null;
    let root = null;
    let svg = null;
    let storeId = null;
    let plots = [];
    let frame = null;
    const plotElements = {};  // node id -> {div, img, naturalAspect}
    const offsets = {};       // node id -> {x, y}, plot centre relative to the node centre, in graph units

    function findCy(graphId) {
        const container = document.getElementById(graphId);
        return container && container._cyreg ? container._cyreg.cy : null;
    }

    function attach(newCy) {
        cy = newCy;
        for (const id of Object.keys(plotElements)) {
            delete plotElements[id];
        }
        root = document.createElement("div");
        root.className = "imote-graph-plots";
        svg = document.createElementNS(SVG_NS, "svg");
        root.appendChild(svg);
        cy.container().appendChild(root);
        cy.on("viewport resize add remove position style data", schedule);
    }

    function render(graphId, newStoreId, newPlots, retry = 0) {
        storeId = newStoreId;
        plots = newPlots;
        const newCy = findCy(graphId);
        if (!newCy) {
            // The graph is not mounted yet, e.g. right after switching pages
            if (retry < MAX_RETRIES) setTimeout(() => render(graphId, newStoreId, newPlots, retry + 1), 100);
            return;
        }
        if (newCy !== cy || !root.isConnected) attach(newCy);

        const ids = new Set(plots.map((plot) => plot.id));
        for (const id of Object.keys(plotElements)) {
            if (!ids.has(id)) {
                plotElements[id].div.remove();
                delete plotElements[id];
                delete offsets[id];
            }
        }
        for (const plot of plots) {
            if (!plotElements[plot.id]) plotElements[plot.id] = createPlotElement(plot.id);
            const element = plotElements[plot.id];
            if (element.img.getAttribute("src") !== plot.src) {
                element.naturalAspect = null;
                element.img.src = plot.src;
            }
        }
        schedule();
    }

    function removePlot(id) {
        const remaining = plots.filter((plot) => plot.id !== id);
        render(cy.container().id, storeId, remaining);
        window.dash_clientside.set_props(storeId, {data: remaining});
    }

    function createPlotElement(id) {
        const div = document.createElement("div");
        div.className = "imote-graph-plot";
        const img = document.createElement("img");
        img.draggable = false;
        img.alt = `Plot of ${id}`;
        const close = document.createElement("button");
        close.className = "imote-graph-plot-close";
        close.title = "Remove this plot from the graph";
        close.innerHTML = '<i class="bi bi-x"></i>';
        div.append(img, close);
        root.appendChild(div);

        const element = {div, img, naturalAspect: null};
        // The svg may be cropped (tight bounding box), so its own size gives the right aspect ratio
        img.addEventListener("load", () => {
            if (img.naturalWidth && img.naturalHeight) element.naturalAspect = img.naturalWidth / img.naturalHeight;
            schedule();
        });

        // Keep cytoscape from starting a pan or a box selection under the plot. Wheel events still go through, so
        // zooming works on top of a plot as well.
        for (const type of ["mousedown", "touchstart", "pointerdown"]) {
            div.addEventListener(type, (event) => event.stopPropagation());
        }
        close.addEventListener("click", (event) => {
            event.stopPropagation();
            removePlot(id);
        });

        div.addEventListener("pointerdown", (event) => {
            if (event.button !== 0 || close.contains(event.target) || !offsets[id]) return;
            event.preventDefault();
            root.appendChild(div);  // bring to the front of overlapping plots
            div.setPointerCapture(event.pointerId);
            div.classList.add("dragging");
            const start = {x: event.clientX, y: event.clientY, offset: {...offsets[id]}};
            let moved = false;

            const onMove = (moveEvent) => {
                const dx = moveEvent.clientX - start.x;
                const dy = moveEvent.clientY - start.y;
                if (Math.hypot(dx, dy) > DRAG_THRESHOLD) moved = true;
                offsets[id] = {x: start.offset.x + dx / cy.zoom(), y: start.offset.y + dy / cy.zoom()};
                schedule();
            };
            const onUp = () => {
                div.classList.remove("dragging");
                div.removeEventListener("pointermove", onMove);
                div.removeEventListener("pointerup", onUp);
                div.removeEventListener("pointercancel", onUp);
                if (!moved) {
                    // A click on the plot selects its node, so the node card shows that node
                    const node = cy.getElementById(id);
                    if (node.nonempty()) {
                        cy.$(":selected").unselect();
                        node.select();
                    }
                }
            };
            div.addEventListener("pointermove", onMove);
            div.addEventListener("pointerup", onUp);
            div.addEventListener("pointercancel", onUp);
        });
        return element;
    }

    function schedule() {
        if (frame === null) frame = requestAnimationFrame(draw);
    }

    function draw() {
        frame = null;
        if (!cy || !root) return;
        svg.replaceChildren();
        const zoom = cy.zoom();
        const pan = cy.pan();
        const toScreen = (x, y) => [x * zoom + pan.x, y * zoom + pan.y];
        const strokeWidth = Math.min(Math.max(1.5 * zoom, 0.75), 3);

        for (const plot of plots) {
            const element = plotElements[plot.id];
            const node = cy.getElementById(plot.id);
            if (node.empty() || !node.visible()) {
                // The node is not in the graph (e.g. collapsed), its plot comes back with the node
                element.div.style.display = "none";
                continue;
            }
            element.div.style.display = "";

            const nodeBox = node.boundingBox({includeLabels: false, includeOverlays: false});
            const position = node.position();
            const width = plot.width;
            const height = width / (element.naturalAspect || plot.aspect);
            if (!offsets[plot.id]) offsets[plot.id] = defaultOffset(node, nodeBox, width);
            const offset = offsets[plot.id];
            const plotBox = {
                x1: position.x + offset.x - width / 2,
                x2: position.x + offset.x + width / 2,
                y1: position.y + offset.y - height / 2,
                y2: position.y + offset.y + height / 2,
            };

            const color = node.data("color") || "#2c3e50";
            const [left, top] = toScreen(plotBox.x1, plotBox.y1);
            Object.assign(element.div.style, {
                left: `${left}px`,
                top: `${top}px`,
                width: `${width * zoom}px`,
                height: `${height * zoom}px`,
                borderColor: color,
            });
            drawConnector(nodeBox, plotBox, color, strokeWidth, toScreen);
        }
    }

    // New plots go next to the node, on the side that points away from the middle of the tree
    function defaultOffset(node, nodeBox, width) {
        const treeBox = cy.nodes().boundingBox({includeLabels: false});
        const side = node.position().x > (treeBox.x1 + treeBox.x2) / 2 ? -1 : 1;
        return {x: side * (nodeBox.w / 2 + GAP + width / 2), y: 0};
    }

    // Square bracket around the node on the side facing the plot, and two lines from its corners to the plot
    function drawConnector(nodeBox, plotBox, color, strokeWidth, toScreen) {
        const gapRight = plotBox.x1 - nodeBox.x2;
        const gapLeft = nodeBox.x1 - plotBox.x2;
        const gapBelow = plotBox.y1 - nodeBox.y2;
        const gapAbove = nodeBox.y1 - plotBox.y2;
        const gap = Math.max(gapRight, gapLeft, gapBelow, gapAbove);
        if (gap <= BRACKET_PAD) return;  // the plot overlaps the node

        // corners: the bracket corner and the plot corner it connects to, for both ends of the bracket
        let corners, armDirection;
        const x1 = nodeBox.x1 - BRACKET_PAD, x2 = nodeBox.x2 + BRACKET_PAD;
        const y1 = nodeBox.y1 - BRACKET_PAD, y2 = nodeBox.y2 + BRACKET_PAD;
        if (gap === gapRight) {
            corners = [[[x2, y1], [plotBox.x1, plotBox.y1]], [[x2, y2], [plotBox.x1, plotBox.y2]]];
            armDirection = [-1, 0];
        } else if (gap === gapLeft) {
            corners = [[[x1, y1], [plotBox.x2, plotBox.y1]], [[x1, y2], [plotBox.x2, plotBox.y2]]];
            armDirection = [1, 0];
        } else if (gap === gapBelow) {
            corners = [[[x1, y2], [plotBox.x1, plotBox.y1]], [[x2, y2], [plotBox.x2, plotBox.y1]]];
            armDirection = [0, -1];
        } else {
            corners = [[[x1, y1], [plotBox.x1, plotBox.y2]], [[x2, y1], [plotBox.x2, plotBox.y2]]];
            armDirection = [0, 1];
        }
        const [[bracketA, plotA], [bracketB, plotB]] = corners;
        const arm = ([x, y]) => [x + armDirection[0] * BRACKET_ARM, y + armDirection[1] * BRACKET_ARM];
        const points = (pts) => pts.map(([x, y]) => toScreen(x, y).join(",")).join(" ");

        // Light wedge between the lines, the "magnifier" area
        addSvg("polygon", {
            points: points([bracketA, plotA, plotB, bracketB]),
            fill: color,
            "fill-opacity": 0.07,
        });
        for (const [from, to] of [[bracketA, plotA], [bracketB, plotB]]) {
            addSvg("polyline", {
                points: points([from, to]),
                stroke: color,
                "stroke-width": strokeWidth,
                "stroke-opacity": 0.8,
            });
        }
        addSvg("polyline", {
            points: points([arm(bracketA), bracketA, bracketB, arm(bracketB)]),
            fill: "none",
            stroke: color,
            "stroke-width": strokeWidth * 1.6,
            "stroke-linejoin": "miter",
        });
    }

    function addSvg(tag, attributes) {
        const shape = document.createElementNS(SVG_NS, tag);
        for (const [name, value] of Object.entries(attributes)) shape.setAttribute(name, value);
        svg.appendChild(shape);
    }

    window.imoteGraphPlots = {render};
})();
