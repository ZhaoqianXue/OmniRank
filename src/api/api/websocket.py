"""
OmniRank WebSocket Handler
Provides real-time streaming of agent messages and progress updates.
"""

import json
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.schemas import WSMessage, WSMessageType, WSProgressPayload

websocket_router = APIRouter(tags=["websocket"])

# Active WebSocket connections per session
active_connections: Dict[str, Set[WebSocket]] = {}


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        self.active_connections[session_id].add(websocket)
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        """Remove a WebSocket connection."""
        if session_id in self.active_connections:
            self.active_connections[session_id].discard(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
    
    async def send_message(self, session_id: str, message: WSMessage):
        """Send a message to all connections in a session."""
        if session_id in self.active_connections:
            message_json = message.model_dump_json()
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_text(message_json)
                except Exception:
                    # Connection might be closed
                    pass
    
    async def broadcast_progress(self, session_id: str, progress: float, message: str):
        """Broadcast a progress update to all connections in a session."""
        ws_message = WSMessage(
            type=WSMessageType.PROGRESS,
            payload={"progress": progress, "message": message},
        )
        await self.send_message(session_id, ws_message)


manager = ConnectionManager()


@websocket_router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time updates.
    
    Clients connect to receive:
    - Progress updates during analysis
    - Agent messages (thinking, actions)
    - Final results or errors
    """
    await manager.connect(websocket, session_id)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "payload": {"session_id": session_id},
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            
            # Parse incoming message
            try:
                message = json.loads(data)
                msg_type = message.get("type")
                
                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                elif msg_type == "subscribe":
                    # Client wants to subscribe to updates
                    await websocket.send_json({
                        "type": "subscribed",
                        "payload": {"session_id": session_id},
                    })
                else:
                    # Echo unknown messages for debugging
                    await websocket.send_json({
                        "type": "echo",
                        "payload": message,
                    })
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "payload": {"message": "Invalid JSON"},
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)


# Export for use in other modules
def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance."""
    return manager
