import asyncio
import logging
import os
import time
import traceback

import websockets.asyncio.server
import websockets.frames

from autobio_inference.base_policy import BasePolicy
from autobio_inference import msgpack_numpy

class WebsocketPolicyServer:
    def __init__(
        self,
        policy: BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        if websocket.request.path == "/switch":
            await self._handle_switch(websocket)

        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        while True:
            try:
                payload = await websocket.recv()
                start = time.perf_counter()
                obs = msgpack_numpy.unpackb(payload)
                action = self._policy.infer(obs)
                response = packer.pack(action)
                await websocket.send(response)
                end = time.perf_counter()
                logging.info(f"Policy inference took {end - start:.2f} seconds")
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    async def _handle_switch(self, websocket: websockets.asyncio.server.ServerConnection):
        payload = await websocket.recv(decode=False)
        arg_file = os.environ["RELAUNCH_ARG_FILE"]
        with open(arg_file, "wb") as f:
            f.write(payload)
        logging.info(f"Relaunching server with args from {arg_file}: {payload}")
        event_loop = asyncio.get_event_loop()
        event_loop.stop()
        exit(101)
