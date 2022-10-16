
from math import sin
from typing import Any

import tornado.web
from plotly import graph_objs
from plotly.io import to_json


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        plot_data = self.prepare_plot_data()
        self.render("index.html", **plot_data)

    def prepare_plot_data(self) -> dict[str, Any]:
        graph = graph_objs.Figure()
        x_data = []
        graph.add_trace(graph_objs.Histogram(x=x_data, xbins={"size": 3}))
        graph.update_layout(
            title="График",
            xaxis_title="X",
            yaxis_title="Y",
        )
        return {"plot": to_json(graph)}
