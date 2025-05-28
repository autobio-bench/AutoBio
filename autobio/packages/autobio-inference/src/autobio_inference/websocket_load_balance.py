import asyncio
import logging
import time
import traceback

import websockets.asyncio.server
import websockets.asyncio.client
import websockets.frames


class WebsocketLoadBalance:
    def __init__(
        self,
        upstreams: list[tuple[str, int]],
        host: str = "0.0.0.0",
        port: int = 16000,
    ) -> None:
        self._upstreams = upstreams
        self._host = host
        self._port = port
        logging.getLogger("websockets.server").setLevel(logging.INFO)

        self._upstream_index = 0
        self._upstream_lock = asyncio.Lock()
    
    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def _get_next_upstream(self):
        host, port = self._upstreams[self._upstream_index]
        self._upstream_index = (self._upstream_index + 1) % len(self._upstreams)
        while True:
            try:
                return await websockets.asyncio.client.connect(
                    f"ws://{host}:{port}",
                    compression=None,
                    max_size=None,
                )
            except ConnectionRefusedError:
                logging.info(f"Connection to {host}:{port} refused. Retrying...")
                await asyncio.sleep(1)
            except asyncio.TimeoutError:
                logging.info(f"Connection to {host}:{port} timed out. Retrying...")
                await asyncio.sleep(1)

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

        while True:
            try:
                payload = await websocket.recv()
                start = time.perf_counter()
                async with await self._get_next_upstream() as upstream:
                    await upstream.send(payload)
                    response = await upstream.recv()
                await websocket.send(response)
                end = time.perf_counter()
                logging.info(f"Load balancing took {end - start:.2f} seconds")
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
        connections = [
            await websockets.asyncio.client.connect(
                f"ws://{host}:{port}/switch",
                compression=None,
                max_size=None,
            )
            for host, port in self._upstreams
        ]
        websockets.asyncio.server.broadcast(connections, payload)
        for conn in connections:
            await conn.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Websocket Load Balancer")
    parser.add_argument(
        "upstreams",
        type=str,
        nargs="+",
        help="List of upstream servers in the format host:port",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the load balancer to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=16000,
        help="Port to bind the load balancer to",
    )
    args = parser.parse_args()
    upstreams = []
    for upstream in args.upstreams:
        host, port = upstream.split(":")
        upstreams.append((host, int(port)))
    logging.basicConfig(level=logging.INFO, force=True)
    load_balancer = WebsocketLoadBalance(
        upstreams=upstreams,
        host=args.host,
        port=args.port,
    )
    load_balancer.serve_forever()
