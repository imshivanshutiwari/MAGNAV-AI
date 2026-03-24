"""WebSocket server for broadcasting navigation state."""
import asyncio
import json
import logging
from typing import Set, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


def _default_serializer(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class NavigationWebSocketServer:
    """WebSocket server that broadcasts navigation states to connected clients."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765,
                 processor=None):
        self.host = host
        self.port = port
        self._processor = processor
        self._clients: Set = set()
        self._server = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False

    async def _handler(self, websocket) -> None:
        """Handle incoming WebSocket connection."""
        self._clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        try:
            async for message in websocket:
                # Echo back any received commands (for future use)
                await websocket.send(json.dumps({"status": "ack", "received": message}))
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info("Client disconnected.")

    async def broadcast(self, data: dict) -> None:
        """Send navigation state to all connected clients.

        Args:
            data: Navigation state dict (must be JSON-serializable)
        """
        if not self._clients:
            return
        payload = json.dumps(data, default=_default_serializer)
        disconnected = set()
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except Exception:
                disconnected.add(ws)
        self._clients -= disconnected

    async def start(self) -> None:
        """Start the WebSocket server."""
        try:
            import websockets
            self._running = True
            self._server = await websockets.serve(self._handler, self.host, self.port)
            logger.info(f"WebSocket server listening on ws://{self.host}:{self.port}")
        except ImportError:
            logger.warning("websockets package not installed; WebSocket server disabled.")

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            logger.info("WebSocket server stopped.")

    async def run_broadcast_loop(self, interval_s: float = 0.1) -> None:
        """Periodically broadcast latest navigation state."""
        while self._running:
            if self._processor is not None:
                state = self._processor.get_latest_state()
                if state is not None:
                    await self.broadcast(state)
            await asyncio.sleep(interval_s)
