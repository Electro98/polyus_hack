
import base64
from abc import ABC
from typing import Any, Callable, Generator, Type

import cv2
import numpy as np

from sockets import BaseWebSocket


class BaseCamera(ABC):
    def frames(self) -> Generator[np.ndarray, None, None]:
        """Should create generator with frames."""
        raise NotImplementedError(f"method 'frames' not implemented in {self.__class__.__name__}")


class CameraOpenCV(BaseCamera):
    video_source = 0

    def __init__(
        self,
        video_source=None,
        loop=False,
    ) -> None:
        """Create simple samera from video_source, can loop if it's video."""
        self.video_source = video_source or self.video_source
        self._loop = loop

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Read frames from video_source and yield them."""
        camera = cv2.VideoCapture(self.video_source)
        if not camera.isOpened():
            raise RuntimeError("could not start camera.")

        while True:
            returned, frame = camera.read()

            if not returned and self._loop:
                camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            if not returned:
                raise RuntimeError("no frame available in video capture")

            yield frame


class WebSocketCameraWrapper:
    """Object that bound together camera and socket for streaming process."""

    def __init__(
        self,
        camera: BaseCamera,
        socket: Type[BaseWebSocket],
        frame_handler: Callable[[np.ndarray], np.ndarray] | None,
        frame_encoder: Callable[[np.ndarray], bytes | str | dict[str, Any]],
    ) -> None:
        self.camera = camera
        self.socket = socket
        self.frame_handler = frame_handler or (lambda frame: frame)
        self.frame_encoder = frame_encoder

    def send_frame_task(self) -> Callable[[], None]:
        """
        Create task that process new frame from camera and send it to socket.
        """
        frame_iterator = self.camera.frames()

        def task():
            new_frame = next(frame_iterator)
            processed_frame = self.frame_handler(new_frame)
            prepared_data = self.frame_encoder(processed_frame)
            self.socket.send_message(
                prepared_data,
                isinstance(prepared_data, bytes),
            )
        return task


class FrameEncoders:
    """Class with implementation of basic image encoders."""

    @staticmethod
    def jpeg_base64(frame: np.ndarray) -> str:
        """Encode image in base64 string with jpeg compression."""
        jpg_img = cv2.imencode('.jpg', frame)
        return base64.b64encode(jpg_img[1]).decode('utf-8')
