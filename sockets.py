from contextlib import suppress

import tornado.websocket


class BaseWebSocket(tornado.websocket.WebSocketHandler):
    clients_pool: set["BaseWebSocket"] = set()

    def open(self):
        self.clients_pool.add(self)

    def on_message(self, message):
        raise NotImplementedError(
            "get message on WS without implemented on_message",
        )

    def on_close(self):
        self.clients_pool.remove(self)

    @classmethod
    def send_message(cls, msg: bytes | str | dict, binary: bool = False):
        for client in cls.clients_pool:
            with suppress(tornado.websocket.WebSocketClosedError):
                client.write_message(msg, binary)

    def check_origin(self, origin):
        return True
        # parsed_origin = urllib.parse.urlparse(origin)
        # return parsed_origin.netloc.endswith(".mydomain.com")
