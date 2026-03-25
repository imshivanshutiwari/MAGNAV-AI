"""Main Dash application for MAGNAV-AI Navigation Ops Center."""
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from dashboard.pages import page1_realtime, page2_sensors, page3_models
from dashboard.pages import page4_evaluation, page5_system
from dashboard.callbacks import nav_callbacks, system_callbacks

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="MAGNAV-AI",
)

# Shared stores
_stores = [
    dcc.Store(id="nav-state-store", data={}),
    dcc.Store(id="history-store", data={"ekf": [], "ukf": [], "pf": [], "fused": []}),
    dcc.Store(id="system-metrics-store", data={}),
    dcc.Interval(id="update-interval", interval=500, n_intervals=0),
]

# Sidebar navigation
_sidebar = dbc.Nav(
    [
        dbc.NavLink("🛰 Real-Time", href="/", active="exact", className="mb-1"),
        dbc.NavLink("📡 Sensors", href="/sensors", active="exact", className="mb-1"),
        dbc.NavLink("🤖 ML Models", href="/models", active="exact", className="mb-1"),
        dbc.NavLink("📊 Evaluation", href="/evaluation", active="exact", className="mb-1"),
        dbc.NavLink("⚙ System", href="/system", active="exact", className="mb-1"),
    ],
    vertical=True,
    pills=True,
    className="flex-column",
)

_header = dbc.Navbar(
    dbc.Container([
        html.Span("🧭 MAGNAV-AI — Navigation Ops Center",
                  className="navbar-brand fw-bold fs-5 text-light"),
    ], fluid=True),
    color="dark",
    dark=True,
    className="mb-0 border-bottom border-secondary",
)

_layout = html.Div([
    dcc.Location(id="url"),
    *_stores,
    _header,
    dbc.Container(
        dbc.Row([
            dbc.Col(
                html.Div(_sidebar, className="p-3 border-end border-secondary",
                         style={"minHeight": "calc(100vh - 56px)", "background": "#1a1a2e"}),
                width=2,
            ),
            dbc.Col(
                html.Div(id="page-content", className="p-3"),
                width=10,
            ),
        ]),
        fluid=True,
    ),
])

app.layout = _layout

# Register page callbacks
page1_realtime.register_callbacks(app)
page2_sensors.register_callbacks(app)
page3_models.register_callbacks(app)
page4_evaluation.register_callbacks(app)
page5_system.register_callbacks(app)
nav_callbacks.register_callbacks(app)
system_callbacks.register_callbacks(app)

# Page routing
from dash.dependencies import Input, Output


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    routes = {
        "/": page1_realtime.get_layout,
        "/sensors": page2_sensors.get_layout,
        "/models": page3_models.get_layout,
        "/evaluation": page4_evaluation.get_layout,
        "/system": page5_system.get_layout,
    }
    fn = routes.get(pathname, page1_realtime.get_layout)
    return fn()


def run_app(host: str = "0.0.0.0", port: int = 8050, debug: bool = False):
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_app(debug=True)
