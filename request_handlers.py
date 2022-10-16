
from typing import Any

import tornado.web
from plotly import graph_objs
from plotly.io import to_json

from itertools import chain
from db_stub import LikeDB


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        plot_data = self.prepare_plot_data()
        self.render("index.html", **plot_data)

    def prepare_plot_data(self) -> dict[str, Any]:
        db = LikeDB()
        graph = graph_objs.Figure()
        x_data = list(chain(*db.get_all()))
        graph.add_trace(graph_objs.Histogram(x=x_data, xbins={"size": 3}))
        graph.update_layout(
            title="Распределение руды",
            xaxis_title="Размер в миллиметрах",
            yaxis_title="Частота",
        )
        return {"plot": to_json(graph)}
