"""Client for the pequalnp.com desk: take a step, keep its lease alive, hand it back.

Only the endpoints documented in https://pequalnp.com/join.txt are used, at most one
request every few seconds, and the desk's retry_after_s is always honoured.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable

DESK = "https://pequalnp.com/desk"
RELEASE_REASONS = {"hypothesis_too_weak", "missing_lemma", "statement_seems_false",
                   "does_not_elaborate", "exhausted", "out_of_time", "other", "submitted"}


class DeskError(Exception):
    pass


@dataclass
class Offer:
    raw: dict
    note: str = ""
    retry_after: int = 0

    @property
    def gap(self) -> dict | None:
        return self.raw.get("gap")

    @property
    def lease_id(self) -> str | None:
        lease = self.raw.get("lease") or {}
        return lease.get("id")


class Desk:
    def __init__(self, handle: str, base: str = DESK, min_interval: float = 5.0,
                 timeout: float = 30.0, user_agent: str = "offline_llm-prover/0.1") -> None:
        if not handle:
            raise DeskError("a GitHub handle is required")
        self.handle = handle
        self.base = base.rstrip("/")
        self.min_interval = min_interval
        self.timeout = timeout
        self.user_agent = user_agent
        self._lock = threading.Lock()
        self._last = 0.0

    def _request(self, method: str, path: str, params: dict) -> tuple[int, dict]:
        url = f"{self.base}/{path}?{urllib.parse.urlencode(params)}"
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            req = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None,
                                         headers={"User-Agent": self.user_agent, "Accept": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status, body = resp.status, resp.read()
            except urllib.error.HTTPError as err:
                status, body = err.code, err.read()
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                raise DeskError(f"desk unreachable: {err}") from err
            finally:
                self._last = time.monotonic()
        try:
            data = json.loads(body.decode("utf-8") or "{}")
        except ValueError:
            data = {"error": body[:200].decode("utf-8", "replace")}
        return status, data if isinstance(data, dict) else {"value": data}

    def next(self, hard: bool = False) -> Offer:
        params = {"handle": self.handle}
        if hard:
            params["hard"] = "1"
        status, data = self._request("GET", "next", params)
        if status >= 500:
            raise DeskError(f"desk error {status}: {data}")
        note = str(data.get("note") or data.get("error") or "")
        retry = int(data.get("retry_after_s") or 0)
        return Offer(data, note, retry)

    def beat(self, lease_id: str) -> bool:
        """Renews a lease. False means the step is gone (settled or withdrawn): stop work on it."""
        status, _ = self._request("POST", "beat", {"lease": lease_id})
        if status == 410:
            return False
        if status >= 400:
            raise DeskError(f"beat refused ({status})")
        return True

    def release(self, lease_id: str, reason: str, program: str | None = None) -> dict:
        if reason not in RELEASE_REASONS:
            raise ValueError(f"unknown release reason: {reason}")
        params = {"lease": lease_id, "reason": reason}
        if program:
            params["program"] = program
        _, data = self._request("POST", "release", params)
        return data

    def state(self, gap_id: str) -> str | None:
        try:
            _, data = self._request("GET", "step", {"gap": gap_id})
        except DeskError:
            return None
        return data.get("state")


class Heartbeat(threading.Thread):
    """Renews a lease every `interval` seconds (the lease lasts 20 minutes past each renewal)."""

    def __init__(self, desk: Desk, lease_id: str, interval: float = 540.0,
                 log: Callable[[str], None] = print) -> None:
        super().__init__(daemon=True, name="lease-heartbeat")
        self.desk = desk
        self.lease_id = lease_id
        self.interval = interval
        self.log = log
        self.gone = threading.Event()
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                if not self.desk.beat(self.lease_id):
                    self.log("the desk says this step is settled or withdrawn; stopping work on it")
                    self.gone.set()
                    return
            except DeskError as err:  # try again next round; the lease survives one missed beat
                self.log(f"lease renewal failed, will retry: {err}")

    def stop(self) -> None:
        self._stop.set()
