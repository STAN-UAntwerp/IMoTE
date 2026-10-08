import dash_bootstrap_components as dbc
from dash import html

from imote import ids, status

_ICONS = {
    status.SUCCESS: "bi-check-circle-fill",
    status.INFO: "bi-info-circle-fill",
    status.WARNING: "bi-exclamation-triangle-fill",
    status.ERROR: "bi-x-circle-fill",
}


def render_status_message(message: dict) -> tuple[list, str]:
    children = [
        html.I(className=f"bi {_ICONS[message['level']]} me-2"),
        html.Span(message["text"], className="flex-grow-1 text-break"),
        html.Small(message["time"], className="ms-3 opacity-75 text-nowrap"),
    ]
    class_name = f"alert alert-{message['level']} d-flex align-items-center flex-grow-1 mb-0 py-1 px-3"
    return children, class_name


def render_status_history(messages: list[dict]):
    if not messages:
        return html.Small("No messages yet.", className="text-muted")
    return html.Ul(
        [
            html.Li(
                [
                    html.I(className=f"bi {_ICONS[message['level']]} text-{message['level']} me-2"),
                    html.Small(message["time"], className="text-muted me-2"),
                    html.Span(message["text"]),
                ],
                className="d-flex align-items-baseline mb-1 small",
            )
            for message in reversed(messages)
        ],
        className="list-unstyled mb-0",
    )


def make_status_bar() -> html.Div:
    children, class_name = render_status_message(status.WELCOME)
    return html.Div(
        [
            html.Div(children, id={"type": ids.STATUS_MESSAGE, "index": 0}, className=class_name, role="status",
                     **{"aria-live": "polite"}),
            dbc.Button(
                html.I(className="bi bi-clock-history"),
                id=ids.STATUS_HISTORY_BUTTON,
                color="light",
                size="sm",
                title="Message history",
                class_name="border",
            ),
            dbc.Popover(
                [
                    dbc.PopoverHeader("Message history"),
                    dbc.PopoverBody(render_status_history([]), id={"type": ids.STATUS_HISTORY_LIST, "index": 0}),
                ],
                target=ids.STATUS_HISTORY_BUTTON,
                trigger="legacy",
                placement="bottom-end",
            ),
        ],
        className="d-flex align-items-stretch gap-2 mb-2",
    )
