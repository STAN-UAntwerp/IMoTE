// Node plots pinned on top of the cytoscape graph.
//
// The plots are not cytoscape elements: they live in an overlay div on top of the graph canvas, so they never take
// part in the dagre layout. The overlay follows the pan, zoom and node positions of the graph. Each plot is connected
// to its node with a square bracket around the node and two lines to the near corners of the plot (a "magnifier").
//
// Dash calls window.imoteGraphPlots.render with the plots in the store (see b_node_callbacks.py), each plot is
// {id: cytoscape node id, src: image url, aspect: width / height, width: optional, set after a resize}.
//
// Sizes and positions are in "normal view" units and are scaled to the view of the node (normal, minimal, tiny),
// so a plot and its bracket keep the same size compared to the tree when the view changes.
(function () {
    const SVG_NS = "http://www.w3.org/2000/svg";
    // In normal view units
    const DEFAULT_PLOT_WIDTH = 400;
    const MIN_PLOT_WIDTH = 50;
    const GAP = 40;          // between a node and its new plot
    const BRACKET_PAD = 5;   // between the node and its bracket
    const BRACKET_ARM = 6;   // length of the bracket ends
    const STROKE_WIDTH = 1.5;
    // Size of the plots in each view compared to the normal view, about the size of the tree compared to the normal
    // view, a bit larger for minimal and tiny where the nodes are large compared to the tree. The node classes are set
    // in g_elements_callbacks.py.
    const VIEW_SCALES = {tiny: 0.2, minimal: 0.4, plots_on: 2.2};
    const DRAG_THRESHOLD = 3;  // screen px, a shorter drag is a click that selects the node
    const MAX_RETRIES = 50;
    // Resize handle on each corner: which way the corner moves when the plot grows, the opposite corner stays put
    const CORNERS = {nw: [-1, -1], ne: [1, -1], sw: [-1, 1], se: [1, 1]};

    let cy = null;
    let root = null;
    let svg = null;
    let storeId = null;
    let plots = [];
    let frame = null;
    const plotElements = {};  // node id -> {div, img, naturalAspect}
    const offsets = {};       // node id -> {x, y}, plot centre relative to the node centre
    const widths = {};        // node id -> width while the plot is being resized

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
                delete widths[id];
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
        for (const corner of Object.keys(CORNERS)) {
            const handle = document.createElement("div");
            handle.className = `imote-graph-plot-resize ${corner}`;
            handle.addEventListener("pointerdown", (event) => startResize(event, id, handle, corner));
            div.appendChild(handle);
        }
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
            if (event.button !== 0 || event.target !== img && event.target !== div || !offsets[id]) return;
            event.preventDefault();
            root.appendChild(div);  // bring to the front of overlapping plots
            div.setPointerCapture(event.pointerId);
            div.classList.add("dragging");
            const start = {x: event.clientX, y: event.clientY, offset: {...offsets[id]}};
            const unitsPerPixel = 1 / (cy.zoom() * viewScale(cy.getElementById(id)));
            let moved = false;

            const onMove = (moveEvent) => {
                const dx = moveEvent.clientX - start.x;
                const dy = moveEvent.clientY - start.y;
                if (Math.hypot(dx, dy) > DRAG_THRESHOLD) moved = true;
                offsets[id] = {x: start.offset.x + dx * unitsPerPixel, y: start.offset.y + dy * unitsPerPixel};
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

    function viewScale(node) {
        for (const [view, scale] of Object.entries(VIEW_SCALES)) {
            if (node.hasClass(view)) return scale;
        }
        return 1;
    }

    function plotSize(plot) {
        const width = widths[plot.id] ?? plot.width ?? DEFAULT_PLOT_WIDTH;
        return {width, height: width / (plotElements[plot.id].naturalAspect || plot.aspect)};
    }

    // Dragging a corner resizes the plot and keeps its aspect ratio, the opposite corner stays where it is
    function startResize(event, id, handle, corner) {
        const plot = plots.find((p) => p.id === id);
        if (event.button !== 0 || !plot || !offsets[id]) return;
        event.preventDefault();
        event.stopPropagation();
        root.appendChild(plotElements[id].div);
        handle.setPointerCapture(event.pointerId);

        const [signX, signY] = CORNERS[corner];
        const start = {x: event.clientX, y: event.clientY, offset: {...offsets[id]}, ...plotSize(plot)};
        const unitsPerPixel = 1 / (cy.zoom() * viewScale(cy.getElementById(id)));
        const aspect = start.width / start.height;
        // Centre of the plot = fixed corner + half the size towards the dragged corner
        const fixedCorner = {
            x: start.offset.x - signX * start.width / 2,
            y: start.offset.y - signY * start.height / 2,
        };

        const onMove = (moveEvent) => {
            const dx = signX * (moveEvent.clientX - start.x) * unitsPerPixel;
            const dy = signY * (moveEvent.clientY - start.y) * unitsPerPixel;
            // Follow whichever direction the mouse moved the most
            const width = Math.max(MIN_PLOT_WIDTH, start.width + dx, (start.height + dy) * aspect);
            widths[id] = width;
            offsets[id] = {x: fixedCorner.x + signX * width / 2, y: fixedCorner.y + signY * width / aspect / 2};
            schedule();
        };
        const onUp = () => {
            handle.removeEventListener("pointermove", onMove);
            handle.removeEventListener("pointerup", onUp);
            handle.removeEventListener("pointercancel", onUp);
            if (widths[id] === undefined) return;
            // Save the new width in the store, so Dash knows it as well
            plots = plots.map((p) => (p.id === id ? {...p, width: widths[id]} : p));
            delete widths[id];
            window.dash_clientside.set_props(storeId, {data: plots});
        };
        handle.addEventListener("pointermove", onMove);
        handle.addEventListener("pointerup", onUp);
        handle.addEventListener("pointercancel", onUp);
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
            const scale = viewScale(node);
            let {width, height} = plotSize(plot);
            if (!offsets[plot.id]) offsets[plot.id] = defaultOffset(node, nodeBox, width, scale);
            const offset = {x: offsets[plot.id].x * scale, y: offsets[plot.id].y * scale};
            width *= scale;
            height *= scale;
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
            const strokeWidth = Math.min(Math.max(STROKE_WIDTH * scale * zoom, 0.75), 3);
            drawConnector(nodeBox, plotBox, color, scale, strokeWidth, toScreen);
        }
    }

    // New plots go next to the node, on the side that points away from the middle of the tree
    function defaultOffset(node, nodeBox, width, scale) {
        const treeBox = cy.nodes().boundingBox({includeLabels: false});
        const side = node.position().x > (treeBox.x1 + treeBox.x2) / 2 ? -1 : 1;
        return {x: side * (nodeBox.w / 2 / scale + GAP + width / 2), y: 0};
    }

    // Square bracket around the node on the side facing the plot, and two lines from its corners to the plot
    function drawConnector(nodeBox, plotBox, color, scale, strokeWidth, toScreen) {
        const pad = BRACKET_PAD * scale;
        const gapRight = plotBox.x1 - nodeBox.x2;
        const gapLeft = nodeBox.x1 - plotBox.x2;
        const gapBelow = plotBox.y1 - nodeBox.y2;
        const gapAbove = nodeBox.y1 - plotBox.y2;
        const gap = Math.max(gapRight, gapLeft, gapBelow, gapAbove);
        if (gap <= pad) return;  // the plot overlaps the node

        // corners: the bracket corner and the plot corner it connects to, for both ends of the bracket
        let corners, armDirection;
        const x1 = nodeBox.x1 - pad, x2 = nodeBox.x2 + pad;
        const y1 = nodeBox.y1 - pad, y2 = nodeBox.y2 + pad;
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
        const armLength = BRACKET_ARM * scale;
        const arm = ([x, y]) => [x + armDirection[0] * armLength, y + armDirection[1] * armLength];
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

    // The plot as a nested <svg> element instead of an <image> with a data url: it stays vector, and converters
    // (svg to pdf for LaTeX, Inkscape, ...) don't all read svg images from data urls
    function inlinePlotSvg(src) {
        const text = decodeURIComponent(src.slice(src.indexOf(",") + 1));
        const plotSvg = new DOMParser().parseFromString(text, "image/svg+xml").documentElement;
        if (!plotSvg.getAttribute("viewBox")) {
            const width = parseFloat(plotSvg.getAttribute("width")), height = parseFloat(plotSvg.getAttribute("height"));
            plotSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);
        }
        return document.importNode(plotSvg, true);
    }

    // Downloads the shown part of the graph as svg, with the pinned plots and their brackets on top
    function downloadSvg(graphId, filename) {
        const graphCy = findCy(graphId);
        if (!graphCy) return;
        const graphSvg = new DOMParser().parseFromString(graphCy.svg(), "image/svg+xml").documentElement;
        // so the svg scales when it is shown at another size, e.g. in a browser or a document
        graphSvg.setAttribute("viewBox", `0 0 ${graphSvg.getAttribute("width")} ${graphSvg.getAttribute("height")}`);

        if (graphCy === cy && root && root.isConnected) {
            draw();  // make sure the overlay matches the graph right now
            // cytoscape draws the svg at the screen's pixel ratio, the overlay is in css pixels
            const scale = Number(graphSvg.getAttribute("width")) / cy.width();
            const overlay = document.createElementNS(SVG_NS, "g");
            overlay.setAttribute("transform", `scale(${scale})`);
            for (const shape of svg.children) overlay.appendChild(shape.cloneNode(true));

            for (const plot of plots) {
                const {div, img} = plotElements[plot.id];
                if (div.style.display === "none") continue;
                const box = {
                    x: parseFloat(div.style.left), y: parseFloat(div.style.top),
                    width: parseFloat(div.style.width), height: parseFloat(div.style.height),
                };
                const background = document.createElementNS(SVG_NS, "rect");
                const image = inlinePlotSvg(img.src);
                for (const [name, value] of Object.entries({...box, rx: 4, fill: "#ffffff"})) {
                    background.setAttribute(name, value);
                }
                for (const [name, value] of Object.entries(box)) image.setAttribute(name, value);
                image.setAttribute("preserveAspectRatio", "xMidYMid meet");
                const border = background.cloneNode();
                border.setAttribute("fill", "none");
                border.setAttribute("stroke", div.style.borderColor);
                border.setAttribute("stroke-width", 2);
                overlay.append(background, image, border);
            }
            graphSvg.appendChild(overlay);
        }

        const blob = new Blob([new XMLSerializer().serializeToString(graphSvg)], {type: "image/svg+xml;charset=utf-8"});
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = filename;
        link.click();
        setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    }

    window.imoteGraphPlots = {render, downloadSvg};
})();
