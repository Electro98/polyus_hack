
from typing import Any
from math import sin
import tornado.web

from plotly.io import to_json
from plotly import graph_objs


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        plot_data = self.prepare_plot_data()
        self.render("index.html", **plot_data)

    def prepare_plot_data(self) -> dict[str, Any]:
        graph = graph_objs.Figure()
        x_data = list(range(-100, 100))
        y_data = list(map(lambda x: sin(x / 2), x_data))
        graph.add_trace(graph_objs.Scatter(x=x_data, y=y_data))
        graph.update_layout(
            title="График",
            xaxis_title="X",
            yaxis_title="Y",
        )
        return {"plot": to_json(graph)}
