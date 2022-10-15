import asyncio

import tornado.web

from camera import CameraOpenCV, FrameEncoders, WebSocketCameraWrapper
from sockets import BaseWebSocket
from utils import Periodic

FRAMES_PER_SECOND = 60


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


def make_app():
    """Create main web application."""
    return tornado.web.Application(
        [
            (r"/", MainHandler),
            (r"/websocket/", BaseWebSocket),
        ],
        websocket_ping_interval=10,
        websocket_ping_timeout=30,
        template_path="templates/",
    )


async def create_camera_task():
    """Create task for camera."""
    camera = CameraOpenCV("src/video.mkv", True)
    wrapper = WebSocketCameraWrapper(
        camera=camera,
        socket=BaseWebSocket,
        frame_handler=None,
        frame_encoder=FrameEncoders.jpeg_base64,
    )
    task_period = 1 / FRAMES_PER_SECOND
    camera_task = Periodic(wrapper.send_frame_task(), task_period)
    await camera_task.start()
    return camera_task


async def main():
    """Start server app and camera task."""
    app = make_app()
    app.listen(8888)
    camera_task = await create_camera_task()
    shutdown_event = asyncio.Event()
    await shutdown_event.wait()
    await camera_task.stop()


if __name__ == "__main__":
    asyncio.run(main())
