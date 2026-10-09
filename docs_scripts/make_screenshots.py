"""Makes the screenshots for the documentation by driving the app in a headless browser."""
import logging
import threading
from pathlib import Path

from playwright.sync_api import Browser, Page, sync_playwright
from werkzeug.serving import make_server

from imote import ids
from imote.app import app
from imote.config import get_initial_graph_info

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets" / "screenshots"
VIEWPORT = {"width": 1600, "height": 1000}
OVERVIEW_SCALE = 3  # 4800 x 3000 pixels for the screenshots of the whole app
CARD_SCALE = 2  # 2x pixels for sharp images on high-dpi screens
SETTLE_MS = 1000  # time for the graph layout animation to finish
ANNOTATION_COLOR = "#1f3fbf"

# Selected node in the Node Information screenshots: the BLIN node at the top right of the default tree
INTERNAL_NODE = "node2"

# Finds the Cytoscape instance of the tree graph, cytoscape keeps a reference to it on its container element
_JS_CY = """
const container = [document.getElementById("%s"), ...document.querySelectorAll("#%s *")].find(el => el._cyreg?.cy);
const cy = container._cyreg.cy;
""" % (ids.CYTOSCAPE_GRAPH, ids.CYTOSCAPE_GRAPH)


def wait_until_idle(page: Page):
    """Waits until Dash has no callbacks running and the graph has finished its layout animation."""
    page.wait_for_load_state("networkidle")
    page.wait_for_function("!document.querySelector('[data-dash-is-loading=\"true\"]')")
    page.wait_for_timeout(SETTLE_MS)


def open_app(browser: Browser, url: str, scale: int = CARD_SCALE) -> Page:
    """Opens the app in a new page, every screenshot starts from the app as it is at startup."""
    page = browser.new_page(viewport=VIEWPORT, device_scale_factor=scale)
    page.goto(url)
    page.wait_for_selector(f"#{ids.CYTOSCAPE_GRAPH} canvas")
    wait_until_idle(page)
    return page


def card(page: Page, inner_id: str):
    """The card that has this id or contains the element with this id."""
    return page.locator(f".card#{inner_id}, .card:has(#{inner_id})")


def open_card(page: Page, toggle_id: str, collapse_id: str):
    page.click(f"#{toggle_id}")
    page.wait_for_selector(f"#{collapse_id}.show")
    wait_until_idle(page)


def click_node(page: Page, node_id: str):
    """Clicks a node of the graph like a user would. The graph is a canvas, so the position comes from Cytoscape."""
    position = page.evaluate(f"""(nodeId) => {{
        {_JS_CY}
        const pos = cy.getElementById(nodeId).renderedPosition();
        const box = container.getBoundingClientRect();
        return {{x: box.left + pos.x, y: box.top + pos.y}};
    }}""", node_id)
    page.mouse.click(position["x"], position["y"])
    wait_until_idle(page)


def deepest_leaf(page: Page) -> str:
    """Id of the lowest leaf in the graph, its global predsplot has the most splits."""
    return page.evaluate(f"""() => {{
        {_JS_CY}
        const leaves = cy.nodes('[node_type = "LeafNode"]').toArray();
        return leaves.reduce((a, b) => (b.position().y > a.position().y ? b : a)).id();
    }}""")


def screenshot_from(page: Page, top_id: str, card_locator, path: Path):
    """Screenshot of the part of a card from the element with id top_id down to the bottom of the card."""
    card_box = card_locator.bounding_box()
    top = page.locator(f"#{top_id}").bounding_box()["y"] - 5
    page.screenshot(path=path, clip={"x": card_box["x"], "y": top, "width": card_box["width"],
                                     "height": card_box["y"] + card_box["height"] - top})


def annotate(page: Page, areas: list[tuple[str, str, str]]):
    """Draws a labeled box around each area, given as (css selector, label, label position).

    Label positions: "center" (in the middle of the box), "inside" (bottom right in the box) or "above" (right, just
    above the box).
    """
    page.evaluate("""([areas, color]) => {
        const positions = {
            center: {left: "50%", top: "50%", transform: "translate(-50%, -50%)"},
            inside: {right: "6px", bottom: "6px"},
            above: {right: "6px", bottom: "calc(100% + 8px)"},  // 4px border + 4px gap
        };
        for (const [selector, label, position] of areas) {
            const box = document.querySelector(selector).getBoundingClientRect();
            const frame = document.createElement("div");
            Object.assign(frame.style, {
                position: "fixed", left: `${box.left}px`, top: `${box.top - 3}px`,
                width: `${box.width}px`, height: `${box.height + 6}px`,
                border: `4px solid ${color}`, borderRadius: "4px", pointerEvents: "none", zIndex: 10000,
            });
            const text = document.createElement("span");
            text.textContent = label;
            Object.assign(text.style, {
                position: "absolute", padding: "2px 5px", borderRadius: "4px", whiteSpace: "nowrap",
                background: "rgba(255, 255, 255, 0.9)", color, font: "bold 22px Arial, sans-serif",
                ...positions[position],
            });
            frame.appendChild(text);
            document.body.appendChild(frame);
        }
    }""", [areas, ANNOTATION_COLOR])


# --- Screenshots -------------------------------------------------------------
def screenshot_overview(browser: Browser, url: str):
    page = open_app(browser, url, OVERVIEW_SCALE)
    page.screenshot(path=OUTPUT_DIR / "app-overview.png")
    # The status bar and the cards column have no id of their own, mark them so the annotation can find them
    page.evaluate(f"""() => {{
        document.querySelector('[id*="{ids.STATUS_MESSAGE}"]').parentElement.dataset.annotate = "status";
        document.getElementById("{ids.HIGHLIGHT_TOGGLE_BUTTON}").closest(".col, [class*='col-']")
            .dataset.annotate = "cards";
    }}""")
    annotate(page, [
        ("[data-annotate=status]", "Status Bar", "above"),
        (f"#{ids.CYTOSCAPE_GRAPH}", "Interactive Tree Graph", "inside"),
        ("[data-annotate=cards]", "Interaction Cards", "inside"),
    ])
    page.screenshot(path=OUTPUT_DIR / "app-overview-annotated.png")
    page.close()


def screenshot_tree_info(browser: Browser, url: str):
    page = open_app(browser, url)
    card(page, ids.TREE_INFO_TEXT).screenshot(path=OUTPUT_DIR / "a-tree-info.png")
    page.close()


def screenshot_node_info(browser: Browser, url: str):
    page = open_app(browser, url)
    node_card = card(page, ids.NODE_INFO_PLOT_SWITCH)
    click_node(page, INTERNAL_NODE)
    node_card.screenshot(path=OUTPUT_DIR / "b-node-info.png")

    # The card with a plot is taller than the window and the cards column scrolls, so make the window tall enough
    # for the whole card to be visible, a clip only shows what is in the window
    page.set_viewport_size({"width": VIEWPORT["width"], "height": 2400})
    wait_until_idle(page)

    # Regression plot of the selected internal node
    page.locator(f"#{ids.NODE_INFO_PLOT_SWITCH}").check()
    page.wait_for_selector(f"#{ids.NODE_INFO_PLOT_CONTAINER} img")
    wait_until_idle(page)
    screenshot_from(page, ids.NODE_INFO_PLOT_CONTAINER, node_card, OUTPUT_DIR / "b-node-info-reg.png")

    # Global predsplot of a leaf
    page.get_by_label("Global predsplot").check()
    click_node(page, deepest_leaf(page))
    page.wait_for_selector(f"#{ids.NODE_INFO_PLOT_CONTAINER} img")
    wait_until_idle(page)
    screenshot_from(page, ids.NODE_INFO_PLOT_CONTAINER, node_card, OUTPUT_DIR / "b-node-info-pred.png")
    page.close()


def screenshot_new_tree(browser: Browser, url: str):
    page = open_app(browser, url)
    open_card(page, ids.NEW_TREE_TOGGLE_BUTTON, ids.NEW_TREE_COLLAPSE)
    card(page, ids.NEW_TREE_TOGGLE_BUTTON).screenshot(path=OUTPUT_DIR / "c-new-tree.png")

    # Choosing "+ New dataset (CSV)" in the dataset dropdown opens the upload dialog
    page.click(f"#{ids.INPUT_DATASET}")
    page.get_by_text("+ New dataset (CSV)").click()
    page.wait_for_selector(f"#{ids.MODAL_CSV} .modal-content")
    wait_until_idle(page)
    # The dialog has rounded corners, hide the app behind it so they show white instead of the page
    page.add_style_tag(content="#react-entry-point { visibility: hidden; } .modal-backdrop { display: none; }")
    page.locator(f"#{ids.MODAL_CSV} .modal-content").screenshot(path=OUTPUT_DIR / "csv_upload.png")
    page.close()


def screenshot_edit_tree(browser: Browser, url: str):
    page = open_app(browser, url)
    open_card(page, ids.EDIT_TREE_TOGGLE_BUTTON, ids.EDIT_TREE_COLLAPSE)
    card(page, ids.EDIT_TREE_TOGGLE_BUTTON).screenshot(path=OUTPUT_DIR / "d-edit-tree.png")
    page.close()


def screenshot_layout(browser: Browser, url: str):
    page = open_app(browser, url)
    open_card(page, ids.LAYOUT_TOGGLE_BUTTON, ids.LAYOUT_COLLAPSE)
    card(page, ids.LAYOUT_TOGGLE_BUTTON).screenshot(path=OUTPUT_DIR / "e-layout.png")
    page.close()


def screenshot_highlight(browser: Browser, url: str):
    page = open_app(browser, url)
    open_card(page, ids.HIGHLIGHT_TOGGLE_BUTTON, ids.HIGHLIGHT_COLLAPSE)
    # A fixed point (the first training row) instead of "Random point", so the screenshot is the same every run
    viz_tree_dict, _ = get_initial_graph_info()
    page.fill(f"#{ids.INPUT_HIGHLIGHT}", ", ".join(map(str, viz_tree_dict["X_train"][0])))
    page.click(f"#{ids.BTN_HIGHLIGHT}")
    page.wait_for_selector(f"#{ids.HIGHLIGHT_RESULT} *")
    wait_until_idle(page)
    card(page, ids.HIGHLIGHT_TOGGLE_BUTTON).screenshot(path=OUTPUT_DIR / "f-highlight.png")
    page.close()


SCREENSHOTS = [
    screenshot_overview,
    screenshot_tree_info,
    screenshot_node_info,
    screenshot_new_tree,
    screenshot_edit_tree,
    screenshot_layout,
    screenshot_highlight,
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)  # no line per request in the terminal
    server = make_server("127.0.0.1", 0, app.server, threaded=True)  # port 0: any free port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for make_screenshot in SCREENSHOTS:
                print(f"{make_screenshot.__name__}...")
                make_screenshot(browser, url)
            browser.close()
    finally:
        server.shutdown()
    print(f"Screenshots saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
