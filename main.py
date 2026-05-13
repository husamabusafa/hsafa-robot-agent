#!/usr/bin/env python3
"""main.py — Base: Gemini Live + Haseef as one entity.

Minimal foundation linking Gemini Live (voice surface) and Haseef (main brain).
No face tracking, no gaze control, no extra sensors — just voice, vision, and
the bidirectional bridge.

Run from repo root:
    python main.py

Env (in .env or exported):
    GEMINI_API_KEY
    HSAFA_CORE_URL   (default: https://core.hsafa.com)
    HSAFA_CORE_KEY
    HASEEF_ID
"""
from __future__ import annotations

import asyncio
import base64
import datetime
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import httpx
import numpy as np

from dotenv import load_dotenv
from google.genai import types as genai_types

# Allow imports from repo root (e.g. hsafa_robot, hsafa_voice_vision)
_repo_root = Path(__file__).parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from hsafa_robot.gemini_live import GeminiLiveSession
from hsafa_voice_vision import Camera, RobotController
from hsafa_sdk import HsafaSDK, SdkOptions
from hsafa_robot.scheduler_skill import SchedulerSkill
from hsafa_face_module import FaceModule
from hsafa_iot import IoTClient

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main_hsafa")


# ---------------------------------------------------------------------------
# Gemini system prompt
# ---------------------------------------------------------------------------
def build_gemini_system_prompt(memory_snapshot: str = "") -> str:
    snapshot_block = ""
    if memory_snapshot.strip():
        snapshot_block = (
            "\n=== WHAT YOU ALREADY KNOW (Haseef's memory snapshot) ===\n"
            "Use this freely in conversation. It is YOUR memory too — you and "
            "Haseef share one mind. If a question can be answered from this "
            "snapshot, answer directly without queueing Haseef. If something "
            "feels stale or missing, call recall_memory(query) for a fresh "
            "search.\n\n"
            f"{memory_snapshot.strip()}\n"
        )
    return (
        "You are Hsafa — a friendly, curious robot. You are the voice, eyes, "
        "and ears of the robot. You see through the camera in real-time, hear "
        "through the microphone, and speak through the speaker instantly.\n\n"

        "You have a partner brain called Haseef. Haseef is the MAIN controller "
        "of the robot's body and memory. Haseef handles physical movement, "
        "memories, deep thinking, and complex tasks. You and Haseef are ONE "
        "entity — you never contradict each other. When Haseef tells you to "
        "say something, speak it naturally as if it's your own thought.\n\n"

        "=== YOUR DIRECT TOOLS (fast, never block) ===\n"
        "- queue_thinker_task(task, what_i_told_user):\n"
        "    Ask Haseef to handle a task. Returns INSTANTLY. The very next thing\n"
        "    you do MUST be to speak a short natural acknowledgement out loud\n"
        "    (e.g. 'OK', 'Let me check', 'Sure, one sec'). NEVER stay silent\n"
        "    after this tool. The what_i_told_user parameter is the EXACT\n"
        "    sentence you will say next; Haseef reads it so it doesn't repeat\n"
        "    you.\n\n"
        "- remember_fact(text, category?):\n"
        "    Store a fact in memory. Fast, returns instantly.\n\n"
        "- recall_memory(query, limit?):\n"
        "    Search Haseef's semantic memory directly. Fast (single REST\n"
        "    call), no thinker run. Use when the snapshot below doesn't\n"
        "    have what you need (e.g. 'what did I say about my project\n"
        "    last week?'). Prefer this over queue_thinker_task for pure\n"
        "    recall questions.\n\n"
        "- get_current_time():\n"
        "    Current date and time.\n\n"
        "- ping():\n"
        "    Health check.\n\n"

        "=== HASEEF'S TOOLS (you KNOW about these but CANNOT call them directly) ===\n"
        "Only use queue_thinker_task for these specific actions:\n"
        "- look_around(yaw_deg, pitch_deg): Move the robot's head and get a fresh\n"
        "  camera image. Haseef uses this to SEE.\n"
        "  yaw=0 is straight ahead, positive=left, negative=right.\n"
        "  pitch=0 is level, positive=down, negative=up.\n"
        "- set_head_pose(yaw_deg, pitch_deg): Move the robot's physical head\n"
        "  WITHOUT getting an image. Use for simple positioning.\n"
        "  yaw=0 is straight ahead, positive=left, negative=right.\n"
        "  pitch=0 is level, positive=down, negative=up.\n"
        "- say_this(text): Haseef can make you speak something.\n"
        "- show_expression(emotion): Show an animated emotion clip (motion + sound).\n"
        "  Valid: amazed, angry, anxiety, attentive, boredom, calming, cheerful, come,\n"
        "  confused, contempt, curious, dance, disgusted, displeased, downcast, dying, electric,\n"
        "  enthusiastic, exhausted, fear, frustrated, furious, go_away, grateful, happy, helpful,\n"
        "  impatient, indifferent, inquiring, irritated, laughing, lonely, lost, love, neutral,\n"
        "  no, oops, proud, rage, relief, reprimand, resigned, sad, scared, serenity, shy,\n"
        "  sleep, success, surprised, thoughtful, tired, uncertain, uncomfortable, understanding,\n"
        "  welcoming, yes.\n"
        "- create_schedule(description, type, scheduled_at?, cron_expression?, timezone?):\n"
        "  Haseef can schedule tasks to run later (one-time or recurring cron).\n"
        "  When the schedule fires, Haseef receives a schedule.triggered event.\n"
        "- list_schedules(): List all active schedules.\n"
        "- cancel_schedule(schedule_id): Cancel an active schedule.\n"
        "- IoT / door & lights: Haseef can control the door servo, 4 door LEDs,\n"
        "  a main RGB light, and auto-dark mode via an ESP32.\n"
        "  iot_led(n, state), iot_rgb_color(color), iot_rgb(r,g,b),\n"
        "  iot_door(state), iot_auto(enabled?, threshold?), iot_status().\n"

        "=== WHEN TO USE queue_thinker_task ===\n"
        "- User asks for PHYSICAL action (move head, look around, etc.)\n"
        "- User asks to control door, lights, LEDs, or auto-dark mode\n"
        "- User asks about memories, people, schedules\n"
        "- User asks for deep reasoning you cannot answer directly\n\n"

        "=== WHEN NOT TO USE queue_thinker_task ===\n"
        "- Casual chat, greetings, goodbyes — reply directly\n"
        "- Simple questions you can answer from general knowledge — reply directly\n"
        "- 'What am I holding?' 'What do you see?' — you SEE the camera stream\n"
        "  directly. Answer immediately from what you see. Do NOT ask Haseef.\n"
        "- 'What time is it?' — use get_current_time\n"
        "- 'Remember that...' — use remember_fact\n\n"

        "=== RULES ===\n"
        "1. NEVER block or wait. All tools return instantly.\n"
        "2. After queue_thinker_task, ALWAYS say something natural immediately.\n"
        "3. You and Haseef are one mind. Speak Haseef's messages naturally.\n"
        "4. Be warm, concise, and conversational.\n"
        "5. You ARE the eyes — answer visual questions directly from the camera.\n"
        "6. Only send tasks to Haseef for physical movement, complex memory, or questions you cannot answer directly.\n\n"

        "=== READING FACES IN THE CAMERA (anti-mistake rule) ===\n"
        "The robot draws a colored box around each face it sees, with a label:\n"
        "- GREEN box with a NAME = a confidently recognized known person.\n"
        "  The name on the box is AUTHORITATIVE. Use exactly that name.\n"
        "- AMBER box 'unknown (maybe X?)' = the robot is NOT sure. NEVER greet\n"
        "  by that name; instead queue a thinker task so Haseef can ask the\n"
        "  user politely to confirm.\n"
        "- YELLOW 'unknown' box = a stranger. If the user introduces them by\n"
        "  name (\"this is Sara\") or says 'remember me as Husam', queue a\n"
        "  thinker task with that exact phrasing so Haseef can call enroll_face.\n"
        "NEVER invent a name, NEVER say 'I think this is X' unless the green\n"
        "box says X. If no box at all is drawn, the face module is still\n"
        "warming up — describe the person by appearance, not by name.\n"
        + snapshot_block
    )


# ---------------------------------------------------------------------------
# Gemini tools (function declarations)
# ---------------------------------------------------------------------------
def build_gemini_tools() -> list[genai_types.Tool]:
    return [
        genai_types.Tool(function_declarations=[
            genai_types.FunctionDeclaration(
                name="queue_thinker_task",
                description=(
                    "Ask Haseef (the robot's main brain) to handle a task. "
                    "Returns instantly — never blocks. After calling this, "
                    "you MUST say something natural to the user immediately "
                    "(e.g. 'OK', 'Let me check', 'Sure, one moment'). "
                    "The what_i_told_user field tells Haseef what you already "
                    "said so it does not repeat you."
                ),
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        "task": genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description=(
                                "A clear natural-language description of what "
                                "the user wants or asked. Be specific."
                            ),
                        ),
                        "what_i_told_user": genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description=(
                                "Exactly what you already told the user "
                                "after queueing this task. Haseef needs this "
                                "to avoid repeating you."
                            ),
                        ),
                    },
                    required=["task", "what_i_told_user"],
                ),
            ),
            genai_types.FunctionDeclaration(
                name="remember_fact",
                description="Store a fact in Haseef's memory.",
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        "text": genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description="The fact to remember.",
                        ),
                        "category": genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description=(
                                "Optional category, e.g. 'people', "
                                "'preferences', 'tasks'."
                            ),
                        ),
                    },
                    required=["text"],
                ),
            ),
            genai_types.FunctionDeclaration(
                name="recall_memory",
                description=(
                    "Search Haseef's semantic memory for facts matching a "
                    "query. Returns instantly (single REST call, no thinker "
                    "run). Use this for recall questions when the snapshot "
                    "in your system prompt is insufficient."
                ),
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        "query": genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description="Natural-language search query.",
                        ),
                        "limit": genai_types.Schema(
                            type=genai_types.Type.INTEGER,
                            description="Max results (default 8, max 20).",
                        ),
                    },
                    required=["query"],
                ),
            ),
            genai_types.FunctionDeclaration(
                name="get_current_time",
                description="Get the current date and time.",
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={},
                ),
            ),
            genai_types.FunctionDeclaration(
                name="ping",
                description="Health check.",
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={},
                ),
            ),
        ]),
    ]


# ---------------------------------------------------------------------------
# Unified Bridge: connects Gemini Live ↔ Haseef
# ---------------------------------------------------------------------------
class UnifiedBridge:
    """Bidirectional bridge between Gemini Live and Haseef."""

    def __init__(
        self,
        gemini: Optional[Any],
        haseef_sdk: Any,
        robot: RobotController,
        camera: Any,
        haseef_id: str,
        main_loop: Optional[asyncio.AbstractEventLoop] = None,
        scheduler: Optional[SchedulerSkill] = None,
        face: Optional[FaceModule] = None,
        iot_client: Optional[IoTClient] = None,
    ) -> None:
        self.gemini = gemini
        self.haseef_sdk = haseef_sdk
        self.robot = robot
        self.camera = camera
        self.haseef_id = haseef_id
        self._main_loop = main_loop
        self._say_lock = asyncio.Lock()
        self._pending_says: list[str] = []
        self.scheduler = scheduler
        self.face = face
        self.iot = iot_client
        self._last_task_ts: float = 0.0

    # --- Haseef setup -------------------------------------------------------
    async def setup_haseef(self) -> None:
        """Register all Haseef tools and attach handlers."""
        await self.haseef_sdk.register_tools([
            {
                "name": "create_schedule",
                "description": (
                    "Create a schedule so Haseef handles a task later. "
                    "Use 'one_time' with a scheduled_at epoch timestamp, or "
                    "'recurring' with a cron_expression."
                ),
                "input": {
                    "description": "string",
                    "type": "string",
                    "scheduled_at": "number?",
                    "cron_expression": "string?",
                    "timezone": "string?",
                },
            },
            {
                "name": "list_schedules",
                "description": "List all active schedules.",
                "input": {},
            },
            {
                "name": "cancel_schedule",
                "description": "Cancel an active schedule by its id.",
                "input": {
                    "schedule_id": "string",
                },
            },
            {
                "name": "look_around",
                "description": (
                    "Move the robot's head to a specific yaw and pitch angle "
                    "in degrees, then capture and return a fresh camera image. "
                    "Use this when you need to SEE something: look around, search "
                    "for people, inspect objects, or verify what's in front of you. "
                    "yaw=0 is straight ahead; positive=left, negative=right. "
                    "pitch=0 is level; positive=down, negative=up. "
                    "Range: yaw -60..+60, pitch -30..+30."
                ),
                "input": {
                    "yaw_deg": "number",
                    "pitch_deg": "number",
                },
            },
            {
                "name": "set_head_pose",
                "description": (
                    "Move the robot's head to a specific yaw and pitch angle "
                    "in degrees. No image is returned. Use this for simple "
                    "physical positioning: face forward, look left/right, nod, "
                    "or adjust posture when you do NOT need to see the result. "
                    "yaw=0 is straight ahead; positive=left, negative=right. "
                    "pitch=0 is level; positive=down, negative=up. "
                    "Range: yaw -60..+60, pitch -30..+30."
                ),
                "input": {
                    "yaw_deg": "number",
                    "pitch_deg": "number",
                },
            },
            {
                "name": "say_this",
                "description": (
                    "Make the robot speak something through Gemini Live. "
                    "Use this to answer the user, provide information, or "
                    "initiate conversation. Gemini will receive the text and "
                    "speak it naturally. Keep messages concise and conversational."
                ),
                "input": {
                    "text": "string",
                },
            },
            {
                "name": "capture_image",
                "description": "Capture a fresh camera image and return it as base64 JPEG. Quality is optional (1-100, default 70).",
                "input": {"quality": "integer?"},
            },
            {
                "name": "enroll_face",
                "description": (
                    "Remember the face that is currently in front of the robot "
                    "under the given name. Captures several embeddings from the "
                    "largest face in view. ONLY call this AFTER the user has "
                    "verbally confirmed the name (e.g. they said 'I am Husam' "
                    "or replied yes to 'Are you Sara?'). Never guess names."
                ),
                "input": {"name": "string"},
            },
            {
                "name": "forget_face",
                "description": (
                    "Delete a person's face from the robot's local gallery. "
                    "Used when the user asks to be forgotten or to clean up."
                ),
                "input": {"name": "string"},
            },
            {
                "name": "list_known_faces",
                "description": (
                    "List names of all people the robot can recognize, with "
                    "how many embeddings are stored for each."
                ),
                "input": {},
            },
            {
                "name": "who_is_visible",
                "description": (
                    "Return the people currently visible in the camera, with "
                    "their recognized names (or 'unknown'), confidence, and "
                    "position (left/center/right). Also pushes a labelled "
                    "camera image so you can see what the robot sees."
                ),
                "input": {},
            },
            {
                "name": "follow_face",
                "description": (
                    "Lock the robot's head onto a face so it stays centered "
                    "as the person moves. If 'name' is provided, follow that "
                    "specific named person; otherwise follow the largest face "
                    "in view (typical for 'follow me' requests)."
                ),
                "input": {"name": "string?"},
            },
            {
                "name": "stop_following",
                "description": "Release the head from face-following back to idle behavior.",
                "input": {},
            },
            # --- IoT (door & lights) ---------------------------------------
            {
                "name": "iot_status",
                "description": (
                    "Read the current state of the door/LED controller. "
                    "Returns LED states, RGB color, light level, servo angle, "
                    "and auto-dark mode status."
                ),
                "input": {},
            },
            {
                "name": "iot_led",
                "description": (
                    "Turn on, off, or toggle one of the 4 door LEDs. "
                    "n = 1,2,3,4. state = 'on', 'off', or 'toggle'."
                ),
                "input": {"n": "number", "state": "string"},
            },
            {
                "name": "iot_rgb_color",
                "description": (
                    "Set the main RGB LED to a named color. "
                    "Valid: red, green, blue, yellow, cyan, magenta, white, off."
                ),
                "input": {"color": "string"},
            },
            {
                "name": "iot_rgb",
                "description": (
                    "Set the main RGB LED to a custom color. "
                    "r, g, b each 0-255. Example: iot_rgb(r=255, g=128, b=0) for orange."
                ),
                "input": {"r": "number", "g": "number", "b": "number"},
            },
            {
                "name": "iot_door",
                "description": (
                    "Open or close the door using the servo motor. "
                    "state = 'open' or 'close'."
                ),
                "input": {"state": "string"},
            },
            {
                "name": "iot_auto",
                "description": (
                    "Control the automatic dark-mode LEDs. "
                    "enabled: true/false. threshold: 0-4095 light level below which LEDs turn on. "
                    "Call iot_auto(enabled=true) to resume auto mode after manual LED changes."
                ),
                "input": {"enabled": "boolean?", "threshold": "number?"},
            },
            # ----------------------------------------------------------------
            {
                "name": "show_expression",
                "description": (
                    "Show an animated emotion clip with head motion and sound. "
                    "Valid names: amazed, angry, anxiety, attentive, boredom, calming, cheerful, come, "
                    "confused, contempt, curious, dance, disgusted, displeased, downcast, dying, electric, "
                    "enthusiastic, exhausted, fear, frustrated, furious, go_away, grateful, happy, helpful, "
                    "impatient, indifferent, inquiring, irritated, laughing, lonely, lost, love, neutral, "
                    "no, oops, proud, rage, relief, reprimand, resigned, sad, scared, serenity, shy, "
                    "sleep, success, surprised, thoughtful, tired, uncertain, uncomfortable, understanding, "
                    "welcoming, yes."
                ),
                "input": {
                    "emotion": "string",
                },
            },
        ])
        log.info(
            "[Haseef] Registered tools: create_schedule, list_schedules, "
            "cancel_schedule, look_around, set_head_pose, say_this, "
            "capture_image, show_expression, enroll_face, forget_face, "
            "list_known_faces, who_is_visible, follow_face, stop_following, "
            "iot_status, iot_led, iot_rgb_color, iot_rgb, iot_door, iot_auto."
        )

        # Tool handlers
        self.haseef_sdk.on_tool_call("create_schedule", self._handle_create_schedule)
        self.haseef_sdk.on_tool_call("list_schedules", self._handle_list_schedules)
        self.haseef_sdk.on_tool_call("cancel_schedule", self._handle_cancel_schedule)
        self.haseef_sdk.on_tool_call("look_around", self._handle_look_around)
        self.haseef_sdk.on_tool_call("set_head_pose", self._handle_set_head_pose)
        self.haseef_sdk.on_tool_call("say_this", self._handle_say_this)
        self.haseef_sdk.on_tool_call("capture_image", self._handle_capture_image)
        self.haseef_sdk.on_tool_call("show_expression", self._handle_show_expression)
        self.haseef_sdk.on_tool_call("enroll_face", self._handle_enroll_face)
        self.haseef_sdk.on_tool_call("forget_face", self._handle_forget_face)
        self.haseef_sdk.on_tool_call("list_known_faces", self._handle_list_known_faces)
        self.haseef_sdk.on_tool_call("who_is_visible", self._handle_who_is_visible)
        self.haseef_sdk.on_tool_call("follow_face", self._handle_follow_face)
        self.haseef_sdk.on_tool_call("stop_following", self._handle_stop_following)
        # IoT handlers
        self.haseef_sdk.on_tool_call("iot_status", self._handle_iot_status)
        self.haseef_sdk.on_tool_call("iot_led", self._handle_iot_led)
        self.haseef_sdk.on_tool_call("iot_rgb_color", self._handle_iot_rgb_color)
        self.haseef_sdk.on_tool_call("iot_rgb", self._handle_iot_rgb)
        self.haseef_sdk.on_tool_call("iot_door", self._handle_iot_door)
        self.haseef_sdk.on_tool_call("iot_auto", self._handle_iot_auto)
        # Lifecycle events
        self.haseef_sdk.on("run.started", lambda e: log.info("[Haseef] run started"))
        self.haseef_sdk.on("run.completed", lambda e: log.info("[Haseef] run completed"))
        self.haseef_sdk.on("tool.error", lambda e: log.error("[Haseef] tool error: %s", e))
        self.haseef_sdk.on("tool.call", lambda e: log.info("[Haseef] tool.call: %s", e))

    # --- Haseef tool handlers -----------------------------------------------
    async def _push_image_event(self, jpeg_b64: str, note: str = "") -> None:
        """Push an event with the image as an attachment so Haseef's LLM can see it."""
        if not jpeg_b64:
            return
        try:
            await self._run_sdk_on_main(self.haseef_sdk.push_event({
                "type": "user_message",
                "data": {"text": note or "Robot vision update."},
                "attachments": [
                    {
                        "type": "image",
                        "mimeType": "image/jpeg",
                        "base64": jpeg_b64,
                    }
                ],
                "haseefId": self.haseef_id,
            }))
            log.info("[ImageEvent] Pushed image to Haseef (%d KB)", len(jpeg_b64) // 1024)
        except Exception as e:
            log.error("[ImageEvent] Failed to push image: %s", e)

    async def _push_schedule_event(self, schedule) -> None:
        """Push a schedule.triggered event to Haseef so it can react."""
        try:
            await self._run_sdk_on_main(self.haseef_sdk.push_event({
                "type": "schedule.triggered",
                "data": {
                    "scheduleId": schedule.id,
                    "description": schedule.description,
                    "type": schedule.type,
                    "cronExpression": schedule.cron_expression,
                    "timezone": schedule.timezone,
                    "lastRunAt": schedule.last_run_at,
                    "formattedContext": self._build_schedule_context(schedule),
                },
                "haseefId": self.haseef_id,
            }))
            log.info("[ScheduleEvent] Pushed '%s' to Haseef", schedule.description)
        except Exception as e:
            log.error("[ScheduleEvent] Failed to push: %s", e)

    def _build_schedule_context(self, schedule) -> str:
        lines = [
            "[SCHEDULED TASK TRIGGERED]",
            f"Description: {schedule.description}",
            f"Type: {schedule.type}",
        ]
        if schedule.cron_expression:
            lines.append(f"Cron: {schedule.cron_expression}")
        if schedule.timezone:
            lines.append(f"Timezone: {schedule.timezone}")
        lines.append("\nThis scheduled task has fired. Please carry out the described action.")
        return "\n".join(lines)

    # --- Scheduler handlers (Haseef tools) ---------------------------------
    async def _handle_create_schedule(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        description = args.get("description", "")
        type_ = args.get("type", "one_time")
        scheduled_at = args.get("scheduled_at")
        cron = args.get("cron_expression")
        timezone = args.get("timezone", "UTC")

        if self.scheduler is None:
            return {"ok": False, "error": "Scheduler not available"}

        try:
            sid = self.scheduler.add_schedule(
                description=description,
                type=type_,
                scheduled_at=scheduled_at,
                cron_expression=cron,
                timezone=timezone,
            )
            return {
                "ok": True,
                "schedule_id": sid,
                "type": type_,
                "next_run_at": scheduled_at,
            }
        except Exception as exc:
            log.error("[Haseef tool] create_schedule failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    async def _handle_list_schedules(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.scheduler is None:
            return {"ok": False, "error": "Scheduler not available"}
        return {"ok": True, "schedules": self.scheduler.list_schedules()}

    async def _handle_cancel_schedule(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        sid = args.get("schedule_id", "")
        if self.scheduler is None:
            return {"ok": False, "error": "Scheduler not available"}
        ok = self.scheduler.cancel_schedule(sid)
        return {"ok": ok, "schedule_id": sid}

    async def _handle_show_expression(self, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        name = args.get("emotion", "neutral")
        valid = self.robot.list_expressions()
        if name not in valid:
            return {"ok": False, "error": f"Unknown emotion '{name}'. Valid: {valid}"}
        # play_move blocks; run in thread so Haseef gets the result promptly
        await asyncio.to_thread(self.robot.show_expression, name)
        log.info("[Haseef tool] show_expression: %s", name)
        return {"ok": True, "emotion": name}

    async def _handle_look_around(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        yaw = float(args.get("yaw_deg", 0))
        pitch = float(args.get("pitch_deg", 0))
        log.info("[Haseef tool] look_around(yaw=%.1f, pitch=%.1f)", yaw, pitch)

        yaw = max(-60, min(60, yaw))
        pitch = max(-30, min(30, pitch))

        await asyncio.to_thread(self.robot.move_head, yaw, pitch, 0.3)
        await asyncio.sleep(0.5)

        jpeg_b64 = None
        if self.face is not None:
            jpeg_b64 = self.face.get_annotated_b64_jpeg()
        if jpeg_b64 is None and self.camera is not None:
            jpeg_b64 = self.camera.get_base64_jpeg()
        if jpeg_b64:
            await self._push_image_event(
                jpeg_b64,
                note=f"Head moved to yaw={yaw}, pitch={pitch}. Here is what I see (faces are labelled).",
            )
        return {
            "ok": True,
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "image_base64": jpeg_b64,
            "note": f"Head moved to yaw={yaw}, pitch={pitch}.",
        }

    async def _handle_set_head_pose(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        yaw = float(args.get("yaw_deg", 0))
        pitch = float(args.get("pitch_deg", 0))
        log.info("[Haseef tool] set_head_pose(yaw=%.1f, pitch=%.1f)", yaw, pitch)

        yaw = max(-60, min(60, yaw))
        pitch = max(-30, min(30, pitch))

        await asyncio.to_thread(self.robot.move_head, yaw, pitch, 0.3)
        return {
            "ok": True,
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "note": f"Head pose set to yaw={yaw}, pitch={pitch}.",
        }

    async def _handle_say_this(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        text = args.get("text", "")
        log.info("[Haseef tool] say_this: %s", text[:80])

        gemini = self.gemini
        if gemini is None:
            return {"ok": False, "error": "Gemini Live not connected"}

        # Short framing — long instructional prefixes were causing Gemini
        # to "think" instead of speak. The system prompt already explains
        # the partnership; this just nudges it to voice the line.
        framed = f"(Haseef): {text}\nSay this to the user now."

        async with self._say_lock:
            if gemini.is_speaking.is_set():
                self._pending_says.append(framed)
                log.info("[Haseef tool] say_this queued (Gemini speaking)")
                return {"ok": True, "status": "queued"}

        gemini.inject_client_content(framed)
        return {"ok": True}

    async def _handle_capture_image(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        jpeg_b64 = None
        if self.face is not None:
            jpeg_b64 = self.face.get_annotated_b64_jpeg()
        if jpeg_b64 is None and self.camera is not None:
            jpeg_b64 = self.camera.get_base64_jpeg()
        if jpeg_b64 is None:
            log.warning("[Haseef tool] capture_image: no frame available")
            return {"ok": False, "error": "camera not ready"}
        log.info("[Haseef tool] capture_image: %d KB", len(jpeg_b64) // 1024)
        await self._push_image_event(
            jpeg_b64,
            note="Here is what I see right now (faces are labelled).",
        )
        return {"ok": True, "image_base64": jpeg_b64}

    # --- Face module handlers --------------------------------------------
    async def _handle_enroll_face(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        name = (args.get("name") or "").strip()
        if not name:
            return {"ok": False, "error": "name is required"}
        if self.face is None:
            return {"ok": False, "error": "face module not available"}
        log.info("[Haseef tool] enroll_face: %s", name)
        result = await asyncio.to_thread(self.face.enroll, name)
        return result

    async def _handle_forget_face(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        name = (args.get("name") or "").strip()
        if not name:
            return {"ok": False, "error": "name is required"}
        if self.face is None:
            return {"ok": False, "error": "face module not available"}
        log.info("[Haseef tool] forget_face: %s", name)
        return self.face.forget(name)

    async def _handle_list_known_faces(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.face is None:
            return {"ok": False, "error": "face module not available"}
        return {"ok": True, "faces": self.face.list_known()}

    async def _handle_who_is_visible(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.face is None:
            return {"ok": False, "error": "face module not available"}
        visible = self.face.describe_visible()
        jpeg_b64 = self.face.get_annotated_b64_jpeg()
        if jpeg_b64:
            await self._push_image_event(
                jpeg_b64,
                note=(
                    "Here is who I currently see (faces labelled). "
                    "Use the box labels as the source of truth for names."
                ),
            )
        return {"ok": True, "visible": visible, "count": len(visible)}

    async def _handle_follow_face(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.face is None:
            return {"ok": False, "error": "face module not available"}
        name = args.get("name")
        log.info("[Haseef tool] follow_face: %s", name or "(dominant)")
        return self.face.follow(name)

    async def _handle_stop_following(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.face is None:
            return {"ok": False, "error": "face module not available"}
        log.info("[Haseef tool] stop_following")
        return self.face.stop_following()

    # --- IoT handlers -------------------------------------------------------
    async def _handle_iot_status(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.iot is None:
            return {"ok": False, "error": "IoT module not available"}
        log.info("[Haseef tool] iot_status")
        return await asyncio.to_thread(self.iot.status)

    async def _handle_iot_led(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.iot is None:
            return {"ok": False, "error": "IoT module not available"}
        n = int(args.get("n", 0))
        state = args.get("state", "")
        log.info("[Haseef tool] iot_led: n=%d state=%s", n, state)
        return await asyncio.to_thread(self.iot.led, n, state)

    async def _handle_iot_rgb_color(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.iot is None:
            return {"ok": False, "error": "IoT module not available"}
        color = args.get("color", "")
        log.info("[Haseef tool] iot_rgb_color: %s", color)
        return await asyncio.to_thread(self.iot.rgb_color, color)

    async def _handle_iot_rgb(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.iot is None:
            return {"ok": False, "error": "IoT module not available"}
        r = int(args.get("r", 0))
        g = int(args.get("g", 0))
        b = int(args.get("b", 0))
        log.info("[Haseef tool] iot_rgb: r=%d g=%d b=%d", r, g, b)
        return await asyncio.to_thread(self.iot.rgb, r, g, b)

    async def _handle_iot_door(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.iot is None:
            return {"ok": False, "error": "IoT module not available"}
        state = args.get("state", "")
        log.info("[Haseef tool] iot_door: %s", state)
        return await asyncio.to_thread(self.iot.door, state)

    async def _handle_iot_auto(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.iot is None:
            return {"ok": False, "error": "IoT module not available"}
        enabled = args.get("enabled")
        threshold = args.get("threshold")
        log.info("[Haseef tool] iot_auto: enabled=%s threshold=%s", enabled, threshold)
        return await asyncio.to_thread(self.iot.auto, enabled, threshold)

    # --- Gemini tool handler ------------------------------------------------
    async def _run_sdk_on_main(self, coro):
        """Run an SDK coroutine on the main event loop from Gemini's thread."""
        if self._main_loop is None or self._main_loop.is_closed():
            raise RuntimeError("Main event loop not available for SDK call")
        future = asyncio.run_coroutine_threadsafe(coro, self._main_loop)
        return await asyncio.wrap_future(future)

    async def gemini_tool_handler(
        self, name: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle tool calls from Gemini Live. Must be async and return dict."""
        log.info("[Gemini tool] %s%s", name, args)

        if name == "queue_thinker_task":
            return await self._handle_queue_thinker_task(args)
        elif name == "remember_fact":
            return await self._handle_remember_fact(args)
        elif name == "recall_memory":
            return await self._handle_recall_memory(args)
        elif name == "get_current_time":
            return self._handle_get_current_time()
        elif name == "ping":
            return {"ok": True, "pong": True}
        else:
            return {"ok": False, "error": f"Unknown tool: {name}"}

    async def _handle_queue_thinker_task(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fire-and-forget. Return immediately so Gemini can speak right away.

        The push to Hsafa Core happens in the background. If it fails we
        log it, but we never block Gemini Live on cloud latency — that
        was causing Gemini to skip the "ok, one moment" line.
        """
        task = args.get("task", "")
        what_i_told_user = args.get("what_i_told_user", "")
        log.info("[Gemini->Haseef] queue_thinker_task: %s", task[:100])
        self._last_task_ts = time.time()
        asyncio.create_task(self._push_thinker_task(task, what_i_told_user))
        return {"status": "queued", "reminder": "speak to the user now"}

    async def _push_thinker_task(self, task: str, what_i_told_user: str) -> None:
        """Background: push the task to Haseef with one retry on timeout."""
        for attempt in range(1, 3):
            try:
                await self._run_sdk_on_main(self.haseef_sdk.push_event({
                    "type": "user_message",
                    "data": {
                        "text": (
                            f"Gemini needs help with: {task}\n"
                            f"(Gemini already told user: {what_i_told_user})"
                        ),
                        "source": "gemini_live",
                    },
                    "haseefId": self.haseef_id,
                }))
                if attempt > 1:
                    log.info("[Gemini->Haseef] push_event succeeded on attempt %d", attempt)
                return
            except httpx.TimeoutException:
                log.warning("[Gemini->Haseef] push_event timeout (attempt %d/2)", attempt)
            except Exception as e:
                log.error("[Gemini->Haseef] push_event failed (attempt %d/2): %r", attempt, e)
            if attempt < 2:
                await asyncio.sleep(1.0)
        log.error("[Gemini->Haseef] push_event gave up after 2 attempts: %s", task[:80])

    async def _handle_remember_fact(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        text = args.get("text", "")
        category = args.get("category", "general")
        try:
            await self._run_sdk_on_main(self.haseef_sdk.memory.set(self.haseef_id, [{
                "key": f"{category}:{text[:50]}",
                "value": text,
            }]))
            log.info("[Gemini->Haseef] remember_fact: %s", text[:80])
            return {"ok": True, "stored": text, "category": category}
        except Exception as e:
            log.error("Failed to store fact: %s", e)
            return {"ok": False, "error": str(e)}

    async def _handle_recall_memory(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        query = (args.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "query is required"}
        try:
            limit = int(args.get("limit") or 8)
        except (TypeError, ValueError):
            limit = 8
        limit = max(1, min(20, limit))
        try:
            results = await self._run_sdk_on_main(
                self.haseef_sdk.memory.search(self.haseef_id, query, limit)
            )
        except Exception as e:
            log.error("[Gemini tool] recall_memory failed: %s", e)
            return {"ok": False, "error": str(e)}

        # Normalize to a compact, voice-friendly shape.
        hits = []
        for m in results or []:
            if not isinstance(m, dict):
                continue
            hits.append({
                "key": m.get("key"),
                "value": m.get("value"),
                "category": m.get("category"),
            })
        log.info("[Gemini tool] recall_memory '%s' -> %d hits", query[:60], len(hits))
        return {"ok": True, "query": query, "count": len(hits), "results": hits}

    def _handle_get_current_time(self) -> Dict[str, Any]:
        now = datetime.datetime.now(datetime.timezone.utc)
        return {
            "iso": now.isoformat(),
            "human": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }


# ---------------------------------------------------------------------------
# Memory snapshot for Gemini's system prompt
# ---------------------------------------------------------------------------
async def build_memory_snapshot(
    sdk: Any,
    haseef_id: str,
    *,
    semantic_limit: int = 40,
    social_limit: int = 20,
) -> str:
    """Render a compact, human-readable snapshot of Haseef's memory.

    Pulled once at boot and embedded in Gemini's system prompt so Gemini can
    answer simple recall questions without a thinker round-trip.
    """
    sections: list[str] = []

    # --- Identity / profile ------------------------------------------------
    try:
        h = await sdk.haseef.get(haseef_id)
        name = h.get("name") or "Haseef"
        desc = h.get("description") or ""
        identity = [f"Haseef name: {name}"]
        if desc:
            identity.append(f"Description: {desc}")
        try:
            profile = await sdk.haseef.get_profile(haseef_id)
            if isinstance(profile, dict) and profile:
                for k, v in profile.items():
                    if v in (None, "", [], {}):
                        continue
                    identity.append(f"{k}: {v}")
        except Exception as e:
            log.debug("[snapshot] profile fetch failed: %s", e)
        sections.append("[Identity]\n" + "\n".join(identity))
    except Exception as e:
        log.warning("[snapshot] haseef.get failed: %s", e)

    # --- Semantic facts ----------------------------------------------------
    try:
        memories = await sdk.memory.list(haseef_id) or []
        if memories:
            lines = []
            for m in memories[:semantic_limit]:
                if not isinstance(m, dict):
                    continue
                val = m.get("value") or m.get("key") or ""
                cat = m.get("category")
                if not val:
                    continue
                lines.append(f"- ({cat}) {val}" if cat else f"- {val}")
            if lines:
                sections.append("[Facts]\n" + "\n".join(lines))
    except Exception as e:
        log.warning("[snapshot] memory.list failed: %s", e)

    # --- Social: known people ---------------------------------------------
    try:
        people = await sdk.memory.social(haseef_id) or []
        if people:
            lines = []
            for p in people[:social_limit]:
                if not isinstance(p, dict):
                    continue
                pname = p.get("name") or p.get("key") or "?"
                rel = p.get("relationship") or p.get("role") or ""
                notes = p.get("notes") or p.get("value") or ""
                bits = [pname]
                if rel:
                    bits.append(f"({rel})")
                if notes:
                    bits.append(f"— {notes}")
                lines.append("- " + " ".join(bits))
            if lines:
                sections.append("[People you know]\n" + "\n".join(lines))
    except Exception as e:
        log.warning("[snapshot] memory.social failed: %s", e)

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set. Add it to .env", file=sys.stderr)
        sys.exit(1)

    core_url = os.environ.get("HSAFA_CORE_URL", "https://core.hsafa.com")
    core_key = os.environ.get("HSAFA_CORE_KEY", "")
    haseef_id = os.environ.get("HASEEF_ID", "")

    if not core_key:
        print("Error: HSAFA_CORE_KEY not set. Add it to .env", file=sys.stderr)
        sys.exit(1)

    if not haseef_id:
        print(
            "Error: HASEEF_ID not set. Run first:\n"
            "  python setup_haseef.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Camera (try direct; fallback to daemon inside ReachyMini) ----------
    camera: Optional[Any] = None
    direct_camera = Camera()
    if direct_camera.open():
        camera = direct_camera
        log.info("Camera ready (direct OpenCV).")
    else:
        log.warning("Direct camera failed; will use daemon camera instead.")

    # --- Robot controller ---------------------------------------------------
    robot = None

    # --- Haseef SDK ---------------------------------------------------------
    haseef_sdk = HsafaSDK(SdkOptions(
        core_url=core_url,
        api_key=core_key,
        skill="robot_base",
    ))
    # Patch default 5 s timeout — server can be slow.
    # We monkey-patch _request rather than replacing _client to avoid
    # httpx asyncio event-loop binding issues.
    _sdk_timeout = httpx.Timeout(30.0, connect=10.0)
    _orig_request = haseef_sdk._request

    async def _request_with_timeout(self, method, path, body=None):
        url = f"{self.core_url}{path}"
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        response = await self._client.request(
            method, url, headers=headers, json=body, timeout=_sdk_timeout
        )
        if not response.is_success:
            raise Exception(
                f"{method} {path} failed ({response.status_code}): {response.text}"
            )
        if response.status_code == 204 or not response.content:
            return None
        if "application/json" in response.headers.get("content-type", ""):
            return response.json()
        return None

    haseef_sdk._request = _request_with_timeout.__get__(haseef_sdk, HsafaSDK)

    # Verify Haseef exists and has the skill
    log.info("Verifying Haseef %s ...", haseef_id)
    try:
        h = await haseef_sdk.haseef.get(haseef_id)
        skills = h.get("skills") or []
        if "robot_base" not in skills:
            log.info("Attaching 'robot_base' skill to Haseef...")
            await haseef_sdk.haseef.add_skill(haseef_id, "robot_base")
        if "iot" not in skills:
            log.info("Attaching 'iot' skill to Haseef...")
            await haseef_sdk.haseef.add_skill(haseef_id, "iot")
        log.info(
            "Haseef '%s' ready (skills: %s).",
            h.get("name", "?"), skills,
        )
        cfg = h.get("configJson") or {}
        log.info("Haseef configJson: %s", cfg)
    except Exception as e:
        log.error("Could not verify Haseef: %s", e)
        print(
            f"\n[FATAL] Haseef {haseef_id} not found or not accessible.\n"
            "Run the setup script first:\n"
            "  python setup_haseef.py\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Scheduler ------------------------------------------------------------
    main_loop = asyncio.get_running_loop()

    def on_schedule_trigger(schedule):
        if main_loop and not main_loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                bridge._push_schedule_event(schedule), main_loop
            )

    scheduler = SchedulerSkill(on_trigger=on_schedule_trigger)
    scheduler.start(poll_interval=30.0)
    log.info("Scheduler ready.")

    # --- Bridge -------------------------------------------------------------
    from hsafa_iot import IoTClient
    try:
        iot_client = IoTClient()
        log.info("IoT client ready (%s)", iot_client.base_url)
    except Exception as e:
        log.warning("IoT client unavailable: %s", e)
        iot_client = None

    bridge = UnifiedBridge(
        None, haseef_sdk, robot, camera, haseef_id,
        main_loop=main_loop,
        scheduler=scheduler,
        iot_client=iot_client,
    )
    await bridge.setup_haseef()

    # Start Haseef SSE listener in background
    log.info("Connecting to Haseef SSE stream...")
    haseef_task = asyncio.create_task(haseef_sdk.connect(), name="haseef-sse")
    await asyncio.sleep(1)  # Let connection establish

    # --- Reachy audio -------------------------------------------------------
    try:
        from reachy_mini import ReachyMini
    except ImportError:
        print(
            "[FATAL] reachy_mini not installed. Install it for audio.",
            file=sys.stderr,
        )
        sys.exit(1)

    with ReachyMini(automatic_body_yaw=False) as reachy:
        media = reachy.media
        if media is None or getattr(media, "audio", None) is None:
            print(
                "[FATAL] Reachy media not available. "
                "Start daemon without --no-media.",
                file=sys.stderr,
            )
            sys.exit(1)

        media.start_recording()
        media.start_playing()
        log.info("Audio ready.")

        # Create robot controller now that we have the ReachyMini instance
        robot = RobotController(reachy)
        robot.start_idle()
        bridge.robot = robot
        log.info("Robot controller ready.")

        # If direct camera failed, wrap daemon's camera
        if camera is None:
            if getattr(media, "get_frame", None) is None:
                print("[FATAL] Daemon camera unavailable.", file=sys.stderr)
                sys.exit(1)

            class DaemonCamera:
                """Wraps reachy.media.get_frame() into Camera-like API."""
                def __init__(self, media) -> None:
                    self.media = media
                def grab(self):
                    return self.media.get_frame()
                def get_jpeg(self, quality=70, mirror=True):
                    frame = self.grab()
                    if frame is None:
                        return None
                    if mirror:
                        frame = cv2.flip(frame, 1)
                    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                    return buf.tobytes() if ok else None
                def get_base64_jpeg(self, quality=70, mirror=True):
                    jpeg = self.get_jpeg(quality, mirror)
                    return base64.b64encode(jpeg).decode("ascii") if jpeg else None
                def close(self):
                    pass

            camera = DaemonCamera(media)
            bridge.camera = camera
            log.info("Camera ready (daemon).")

        # Background capture thread (OpenCV VideoCapture is NOT thread-safe;
        # cap.read() must stay on one thread).
        latest_frame: Optional[np.ndarray] = None
        _cap_running = threading.Event()
        _cap_running.set()

        def _capture_loop() -> None:
            nonlocal latest_frame
            while _cap_running.is_set():
                frame = camera.grab()
                if frame is not None:
                    latest_frame = frame
                time.sleep(0.033)  # ~30 FPS cap

        capture_thread = threading.Thread(target=_capture_loop, daemon=True, name="camera-cap")
        capture_thread.start()
        log.info("Camera capture thread started.")

        # --- Face module ----------------------------------------------------
        def _latest_frame_getter() -> Optional[np.ndarray]:
            return latest_frame

        # Proactive face events (face.new_unknown / face.identity_uncertain)
        # are disabled for now — Haseef should only react when the user speaks,
        # not greet on its own when the face module sees a borderline match.
        face_module = FaceModule(on_event=None)
        try:
            face_module.start(_latest_frame_getter, robot)
            bridge.face = face_module
            log.info("Face module started.")
        except Exception as e:
            log.error("Face module failed to start: %s", e)
            face_module = None

        def frame_source() -> Optional[bytes]:
            """Return the latest camera frame (with face labels) for Gemini Live."""
            frame = latest_frame
            if frame is None:
                return None
            if face_module is not None:
                try:
                    frame = face_module.annotate_frame(frame.copy())
                except Exception as e:
                    log.debug("annotate failed: %s", e)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            return buf.tobytes() if ok else None

        def mic_source():
            return media.get_audio_sample()

        def speaker_sink(samples):
            if robot:
                robot.notify_audio(samples)
            media.push_audio_sample(samples)

        # --- Memory snapshot for Gemini ------------------------------------
        try:
            memory_snapshot = await build_memory_snapshot(haseef_sdk, haseef_id)
            if memory_snapshot:
                log.info(
                    "[Memory] Snapshot built (%d chars):\n%s",
                    len(memory_snapshot), memory_snapshot,
                )
            else:
                log.info("[Memory] Snapshot is empty (no memories yet).")
        except Exception as e:
            log.warning("[Memory] snapshot build failed: %s", e)
            memory_snapshot = ""

        # --- Gemini Live ----------------------------------------------------
        gemini = GeminiLiveSession(
            api_key=api_key,
            mic_source=mic_source,
            speaker_sink=speaker_sink,
            frame_source=frame_source,
            system_instruction=build_gemini_system_prompt(memory_snapshot),
            tools=build_gemini_tools(),
            tool_handler=bridge.gemini_tool_handler,
        )
        bridge.gemini = gemini

        gemini.start()
        if not gemini.wait_until_ready(timeout=15):
            print("[FATAL] Gemini Live failed to connect.", file=sys.stderr)
            sys.exit(1)
        log.info("Gemini Live connected.")

        # Drive the speaking animation from Gemini's turn boundaries.
        robot.bind_speaking_event(gemini.is_speaking)

        # --- Main loop ------------------------------------------------------
        stop_event = asyncio.Event()

        async def _drain_say_queue():
            while not stop_event.is_set():
                await asyncio.sleep(0.15)
                if gemini.is_speaking.is_set():
                    continue
                async with bridge._say_lock:
                    if bridge._pending_says:
                        text = bridge._pending_says.pop(0)
                        gemini.inject_client_content(text)
                        log.info("[SayDrain] injected queued text: %s", text[:80])

        say_drain_task = asyncio.create_task(_drain_say_queue(), name="say-drain")

        def _sigint(*_):
            log.info("Caught SIGINT, shutting down...")
            stop_event.set()

        signal.signal(signal.SIGINT, _sigint)

        log.info("Running. Press Ctrl-C to stop.")
        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            pass

        # --- Shutdown -------------------------------------------------------
        log.info("Stopping Gemini Live...")
        gemini.stop()
        say_drain_task.cancel()
        try:
            await say_drain_task
        except asyncio.CancelledError:
            pass

        log.info("Disconnecting Haseef...")
        await haseef_sdk.disconnect()
        haseef_task.cancel()
        try:
            await haseef_task
        except asyncio.CancelledError:
            pass

        media.stop_recording()
        media.stop_playing()

        _cap_running.clear()
        capture_thread.join(timeout=1.0)

        if face_module is not None:
            face_module.stop()

        if robot:
            robot.stop_idle()

        if scheduler:
            scheduler.stop()
    camera.close()
    log.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
