"""hsafa_iot.py — HTTP client for the ESP32 door/LED controller.

The ESP32 exposes a simple REST API on the local network. All calls are
synchronous GET requests (the device is on the same LAN, latency is < 10 ms).
"""

import logging
import os
from typing import Any, Dict, Optional

import httpx

log = logging.getLogger("iot_client")

DEFAULT_TIMEOUT = 5.0


class IoTClient:
    """Talk to the ESP32 IoT controller."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        raw = base_url or os.environ.get("IOT_ESP_IP", "")
        if not raw:
            raise RuntimeError(
                "IOT_ESP_IP not set. Add it to .env, e.g. IOT_ESP_IP=http://192.168.1.42"
            )
        self.base_url = raw.rstrip("/")
        if not self.base_url.startswith("http"):
            self.base_url = f"http://{self.base_url}"
        self._client = httpx.Client(timeout=DEFAULT_TIMEOUT)
        log.info("[IoT] target: %s", self.base_url)

    # ---- low-level ---------------------------------------------------------
    def _get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            r = self._client.get(url, params=params)
            r.raise_for_status()
            body = r.json()
            body["ok"] = True
            return body
        except Exception as e:
            log.warning("IoT request failed: %s (%s)", url, e)
            return {"ok": False, "error": str(e)}

    # ---- public API --------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        """Read the full device state."""
        return self._get("/status")

    def led(self, n: int, state: str) -> Dict[str, Any]:
        """Control one of the 4 regular LEDs.

        :param n: LED number (1–4).
        :param state: 'on', 'off', or 'toggle'.
        """
        if n not in (1, 2, 3, 4):
            return {"ok": False, "error": "LED number must be 1–4"}
        state = state.lower().strip()
        if state not in ("on", "off", "toggle"):
            return {"ok": False, "error": "state must be on, off, or toggle"}
        return self._get("/led", {"n": n, "state": state})

    def rgb_color(self, color: str) -> Dict[str, Any]:
        """Set the RGB LED by named color.

        Valid names: red, green, blue, yellow, cyan, magenta, white, off.
        """
        valid = {"red", "green", "blue", "yellow", "cyan", "magenta", "white", "off"}
        c = color.lower().strip()
        if c not in valid:
            return {
                "ok": False,
                "error": f"color must be one of: {', '.join(sorted(valid))}",
            }
        return self._get("/rgb", {"color": c})

    def rgb(self, r: int, g: int, b: int) -> Dict[str, Any]:
        """Set the RGB LED by raw 0-255 values."""
        for val, name in ((r, "r"), (g, "g"), (b, "b")):
            if not 0 <= val <= 255:
                return {"ok": False, "error": f"{name} must be 0–255"}
        return self._get("/rgb", {"r": r, "g": g, "b": b})

    def servo(self, angle: int) -> Dict[str, Any]:
        """Move the servo to an angle (0–180)."""
        if not 0 <= angle <= 180:
            return {"ok": False, "error": "angle must be 0–180"}
        return self._get("/servo", {"angle": angle})

    def door(self, state: str) -> Dict[str, Any]:
        """Open or close the door (servo motor).

        :param state: 'open' or 'close'.
        """
        state = state.lower().strip()
        if state == "open":
            return self.servo(90)
        elif state == "close":
            return self.servo(0)
        return {"ok": False, "error": "door state must be 'open' or 'close'"}

    def sensor(self) -> Dict[str, Any]:
        """Read the photoresistor."""
        return self._get("/sensor")

    def auto(
        self, enabled: Optional[bool] = None, threshold: Optional[int] = None
    ) -> Dict[str, Any]:
        """Control the automatic dark-mode feature.

        :param enabled: True to turn auto-dark on, False to turn off.
        :param threshold: Light level (0–4095) below which LEDs auto-turn on.
        """
        params: Dict[str, Any] = {}
        if enabled is not None:
            params["enabled"] = "true" if enabled else "false"
        if threshold is not None:
            if not 0 <= threshold <= 4095:
                return {"ok": False, "error": "threshold must be 0–4095"}
            params["threshold"] = threshold
        return self._get("/auto", params)
