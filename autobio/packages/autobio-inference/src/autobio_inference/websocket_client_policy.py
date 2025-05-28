import logging
import time
import json

import websockets.sync.client

from autobio_inference.base_policy import BasePolicy
from autobio_inference import msgpack_numpy


class WebsocketClientPolicy(BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws = self._wait_for_server()

    def _wait_for_server(self) -> websockets.sync.client.ClientConnection:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                return websockets.sync.client.connect(self._uri, compression=None, max_size=None)
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def infer(self, obs: dict) -> dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self) -> None:
        pass

def switch_policy(args: list[str], host: str = "0.0.0.0", port: int = 8000):
    uri = f"ws://{host}:{port}/switch"
    logging.info(f"Switching policy at {uri}: {args}")
    with websockets.sync.client.connect(uri) as ws:
        ws.send(json.dumps(args))
