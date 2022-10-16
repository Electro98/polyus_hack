from contextlib import suppress

import tornado.websocket


class BaseWebSocket(tornado.websocket.WebSocketHandler):
    """Basic implementation of websocket with send_message func."""

    clients_pool: set["BaseWebSocket"] = set()

    def open(self, *args: str, **kwargs: str):
        """Add client to pool when connection opened."""
        self.clients_pool.add(self)

    def on_message(self, message):
        """Message handler stub."""
        raise NotImplementedError(
            "get message on WS without implemented on_message",
        )

    def on_close(self):
        """Remove client from pool when connection closed."""
        self.clients_pool.remove(self)

    @classmethod
    def send_message(cls, msg: bytes | str | dict, binary: bool = False):
        """Send message to all current clients."""
        for client in cls.clients_pool:
            with suppress(tornado.websocket.WebSocketClosedError):
                client.write_message(msg, binary)

    def check_origin(self, origin):
        """Check origin function should check something."""
        return True
        # parsed_origin = urllib.parse.urlparse(origin)
        # return parsed_origin.netloc.endswith(".mydomain.com")


class BidirectionalSocket(BaseWebSocket):
    oversize_rock = 200

    def on_message(self, message):
        """Message handler stub."""
        try:
            self.__class__.oversize_rock = int(message)
        except:
            pass
