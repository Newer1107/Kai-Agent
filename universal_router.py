from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib import request

from openai import OpenAI

from agent_loop import run_agent
from agent_state import estimate_step_budget

try:
    import pyautogui
except Exception:  # pragma: no cover
    pyautogui = None

try:
    import pyperclip
except Exception:  # pragma: no cover
    pyperclip = None

try:
    from send2trash import send2trash
except Exception:  # pragma: no cover
    send2trash = None


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    method: str
    output: str
    error: str | None
    duration_ms: float


class UniversalRouter:
    def __init__(self, status_callback: Callable[[str], None] | None = None) -> None:
        self._status_callback = status_callback

    def _status(self, message: str) -> None:
        if self._status_callback is not None:
            try:
                self._status_callback(message)
            except Exception:
                pass

    def execute(self, raw_command: str) -> ExecutionResult:
        started = time.perf_counter()
        command = (raw_command or "").strip()
        if not command:
            return ExecutionResult(
                success=False,
                method="fast_path",
                output="",
                error="Command is empty.",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

        try:
            print(f"[ROUTER] Command: '{command}'")
            fast = self.match_fast_path(command)
            if fast is not None:
                if fast[0]:
                    print(f"[ROUTER] Layer 1 SUCCESS: {fast[1]}")
                    return ExecutionResult(
                        success=True,
                        method="fast_path",
                        output=fast[1],
                        error=fast[2],
                        duration_ms=(time.perf_counter() - started) * 1000.0,
                    )
                print(f"[ROUTER] Layer 1 FAILED: {fast[2]} - escalating")
                self._status(f"[ROUTER] Layer 1 matched but failed: {fast[2]} - escalating to Layer 2")
            else:
                print("[ROUTER] Layer 1 NO MATCH - escalating to Layer 2")
                self._status("[ROUTER] Layer 1 no match - escalating to Layer 2")

            if self.ollama_is_available():
                print("[ROUTER] Layer 2 attempting code-gen...")
                code_gen_result = self.run_code_gen(command)
                if code_gen_result.success:
                    print("[ROUTER] Layer 2 SUCCESS")
                    code_gen_result = ExecutionResult(
                        success=True,
                        method="code_gen",
                        output=code_gen_result.output,
                        error=None,
                        duration_ms=(time.perf_counter() - started) * 1000.0,
                    )
                    return code_gen_result
                print(f"[ROUTER] Layer 2 FAILED: {code_gen_result.error}")
                self._status(f"[ROUTER] Layer 2 failed: {code_gen_result.error}")
            else:
                print("[ROUTER] Layer 2 SKIPPED: Ollama not reachable")
                self._status("[ROUTER] Layer 2 skipped: Ollama not reachable")

            print("[ROUTER] Layer 3 vision fallback starting")
            self._status("[ROUTER] Layer 3 vision fallback starting")
            vision = self.run_vision_agent(command)
            return ExecutionResult(
                success=vision.success,
                method="vision_fallback",
                output=vision.output,
                error=vision.error,
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                method="router_guard",
                output="",
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

    def match_fast_path(self, command: str) -> tuple[bool, str, str | None] | None:
        lowered = (command or "").strip().lower()

        # Explicit compound: "open {site} and search {query}"
        open_search_match = re.match(r"^open\s+(\w+)\s+and\s+search(?:\s+for)?\s+(.+)$", lowered)
        if open_search_match:
            site = open_search_match.group(1).strip()
            query_raw = open_search_match.group(2).strip()
            query_enc = urllib.parse.quote_plus(query_raw)
            search_url_map = {
                "youtube": f"https://www.youtube.com/results?search_query={query_enc}",
                "google": f"https://www.google.com/search?q={query_enc}",
                "reddit": f"https://www.reddit.com/search/?q={query_enc}",
                "amazon": f"https://www.amazon.in/s?k={query_enc}",
                "github": f"https://github.com/search?q={query_enc}",
                "twitter": f"https://twitter.com/search?q={query_enc}",
                "flipkart": f"https://www.flipkart.com/search?q={query_enc}",
                "maps": f"https://www.google.com/maps/search/{query_enc}",
            }

            url = search_url_map.get(site)
            if url:
                webbrowser.open(url, new=2)
                return True, f"Opened {site} and searched for: {query_raw}", None
            webbrowser.open(f"https://www.google.com/search?q={query_enc}+site:{site}.com", new=2)
            return True, f"Searched {site} for: {query_raw}", None

        parts = self._split_compound(command)
        if len(parts) > 1:
            results: list[tuple[bool, str, str | None]] = []
            for i, part in enumerate(parts):
                sub_result = self._match_single_fast_path(part)
                if sub_result is None:
                    return None
                results.append(sub_result)
                if sub_result[0] and i < len(parts) - 1:
                    time.sleep(1.5)

            failures = [result for result in results if not result[0]]
            if failures:
                return failures[0]
            return results[-1]

        return self._match_single_fast_path(command)

    def _split_compound(self, command: str) -> list[str]:
        parts = re.split(r"\s+(?:and\s+then|and|then)\s+", command, flags=re.IGNORECASE)
        return [part.strip() for part in parts if part.strip()]

    def _match_single_fast_path(self, command: str) -> tuple[bool, str, str | None] | None:
        lowered = command.lower()

        if any(token in lowered for token in ["http://", "https://"]) or lowered.startswith("open www"):
            url = self._extract_url(command)
            if not url:
                return False, "", "Could not parse URL from command."
            webbrowser.open(url, new=2)
            return True, f"Opened {url}", None

        open_site_match = re.match(r"^open\s+(\w+)$", lowered)
        if open_site_match:
            known_websites = {
                "youtube": "https://www.youtube.com",
                "gmail": "https://mail.google.com",
                "github": "https://github.com",
                "reddit": "https://www.reddit.com",
                "twitter": "https://twitter.com",
                "instagram": "https://www.instagram.com",
                "whatsapp": "https://web.whatsapp.com",
                "netflix": "https://www.netflix.com",
                "amazon": "https://www.amazon.in",
                "flipkart": "https://www.flipkart.com",
                "linkedin": "https://www.linkedin.com",
                "stackoverflow": "https://stackoverflow.com",
                "wikipedia": "https://en.wikipedia.org",
                "maps": "https://maps.google.com",
                "drive": "https://drive.google.com",
                "meet": "https://meet.google.com",
                "notion": "https://notion.so",
                "figma": "https://figma.com",
                "vercel": "https://vercel.com",
            }
            site = open_site_match.group(1).strip()
            if site in known_websites:
                webbrowser.open(known_websites[site], new=2)
                return True, f"Opened {site} in browser.", None

        if lowered.startswith("open "):
            target = lowered.replace("open ", "", 1).strip()
            app_result = self._open_known_app(target)
            if app_result is not None:
                return app_result

        create_file_match = re.match(r"^(create|make) file\s+(.+)$", lowered)
        if create_file_match:
            raw_path = command.split(maxsplit=2)[2].strip().strip('"')
            safe_path = self._safe_workspace_path(raw_path)
            if safe_path is None:
                return False, "", "Refused unsafe file path."
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            safe_path.touch(exist_ok=True)
            return True, f"Created file {safe_path}", None

        create_dir_match = re.match(r"^(create|make) (folder|dir|directory)\s+(.+)$", lowered)
        if create_dir_match:
            raw_path = command.split(maxsplit=3)[3].strip().strip('"')
            safe_path = self._safe_workspace_path(raw_path)
            if safe_path is None:
                return False, "", "Refused unsafe directory path."
            safe_path.mkdir(parents=True, exist_ok=True)
            return True, f"Created folder {safe_path}", None

        delete_file_match = re.match(r"^(delete|remove) file\s+(.+)$", lowered)
        if delete_file_match:
            raw_path = command.split(maxsplit=2)[2].strip().strip('"')
            safe_path = self._safe_workspace_path(raw_path)
            if safe_path is None:
                return False, "", "Refused unsafe delete path."
            if not safe_path.exists():
                return False, "", f"File not found: {safe_path}"
            if send2trash is None:
                return False, "", "send2trash is unavailable; refusing hard delete."
            send2trash(str(safe_path))
            return True, f"Moved to recycle bin: {safe_path}", None

        if lowered.startswith("copy ") and " to clipboard" in lowered:
            text = re.sub(r"\s+to clipboard\s*$", "", command[5:], flags=re.IGNORECASE).strip()
            if pyperclip is None:
                return False, "", "pyperclip is unavailable."
            pyperclip.copy(text)
            return True, "Copied text to clipboard.", None

        if lowered in {"paste clipboard", "paste from clipboard"}:
            if pyperclip is None or pyautogui is None:
                return False, "", "Clipboard/type runtime unavailable."
            pyautogui.typewrite(pyperclip.paste(), interval=0.005)
            return True, "Typed clipboard contents.", None

        if lowered.startswith("type "):
            text = command[5:].strip()
            if not text:
                return False, "", "No text to type."
            if pyautogui is None:
                return False, "", "pyautogui is unavailable."
            pyautogui.typewrite(text[:500], interval=0.005)
            return True, "Typed provided text.", None

        if lowered in {"time", "what time is it", "current time"}:
            return True, time.strftime("%H:%M:%S"), None

        search_on_match = re.match(r"^search\s+(.+?)\s+on\s+(\w+)$", lowered)
        if search_on_match:
            query_raw = search_on_match.group(1).strip()
            site = search_on_match.group(2).strip()
            query_enc = urllib.parse.quote_plus(query_raw)
            site_urls = {
                "youtube": f"https://www.youtube.com/results?search_query={query_enc}",
                "google": f"https://www.google.com/search?q={query_enc}",
                "reddit": f"https://www.reddit.com/search/?q={query_enc}",
                "amazon": f"https://www.amazon.in/s?k={query_enc}",
                "github": f"https://github.com/search?q={query_enc}",
                "twitter": f"https://twitter.com/search?q={query_enc}",
                "wikipedia": f"https://en.wikipedia.org/wiki/Special:Search?search={query_enc}",
            }
            url = site_urls.get(site, f"https://www.google.com/search?q={query_enc}+site:{site}")
            webbrowser.open(url, new=2)
            return True, f"Searching {site} for: {query_raw}", None

        search_match = re.match(r"^search\s+(.+)$", lowered)
        if search_match:
            query_raw = search_match.group(1).strip()
            query = urllib.parse.quote_plus(query_raw)
            webbrowser.open(f"https://www.google.com/search?q={query}", new=2)
            return True, f"Searching Google for: {query_raw}", None

        play_match = re.match(r"^play\s+(.+?)(?:\s+on\s+youtube)?$", lowered)
        if play_match:
            query_raw = play_match.group(1).strip()
            query_enc = urllib.parse.quote_plus(query_raw)
            webbrowser.open(f"https://www.youtube.com/results?search_query={query_enc}", new=2)
            return True, f"Opening YouTube search for: {query_raw}", None

        return None

    def run_code_gen(self, command: str) -> ExecutionResult:
        model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b")
        client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
        )
        prompt = (
            "Return Python code only. The script must be short, deterministic, and safe. "
            "Never delete files, never run shell destructive commands, and stay in current directory. "
            f"Task: {command}"
        )

        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=300,
                timeout=30.0,
                messages=[
                    {"role": "system", "content": "You are a safe automation coder. Return code only."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = (response.choices[0].message.content or "").strip()
            code = self._extract_code_block(text)
            if not code:
                return ExecutionResult(False, "code_gen", "", "Model returned empty code.", 0.0)
            danger = self._dangerous_code_reason(code)
            if danger:
                return ExecutionResult(False, "code_gen", "", f"Rejected generated code: {danger}", 0.0)

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py", encoding="utf-8") as tmp:
                    tmp.write(code)
                    tmp_path = Path(tmp.name)

                completed = subprocess.run(
                    [sys.executable, str(tmp_path)],
                    capture_output=True,
                    text=True,
                    timeout=30.0,
                    cwd=str(Path.cwd()),
                )
                output = (completed.stdout or "").strip()
                if completed.returncode != 0:
                    err = (completed.stderr or "").strip() or f"exit code {completed.returncode}"
                    return ExecutionResult(False, "code_gen", output, err, 0.0)
                return ExecutionResult(True, "code_gen", output or "Code executed.", None, 0.0)
            finally:
                if tmp_path is not None:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
        except subprocess.TimeoutExpired:
            return ExecutionResult(False, "code_gen", "", "Generated code timed out after 10s.", 0.0)
        except Exception as exc:
            return ExecutionResult(False, "code_gen", "", f"{type(exc).__name__}: {exc}", 0.0)

    def run_vision_agent(self, command: str) -> ExecutionResult:
        open_first_match = re.match(r"^open\s+(\w+)\s+(?:and|then)\s+(.+)$", command.strip().lower())
        if open_first_match:
            app_or_site = open_first_match.group(1)
            remaining_goal = open_first_match.group(2)
            open_result = self._match_single_fast_path(f"open {app_or_site}")
            if open_result and open_result[0]:
                message = f"[ROUTER] Opened {app_or_site}, waiting for load..."
                print(message)
                self._status(message)
                time.sleep(2.5)
                command = remaining_goal

        budget = estimate_step_budget(command)
        try:
            step_results = run_agent(goal=command, max_steps=budget, pause_seconds=0.2)
        except Exception as exc:
            return ExecutionResult(False, "vision_fallback", "", f"{type(exc).__name__}: {exc}", 0.0)

        executed_steps = sum(1 for step in step_results if step.executed)
        final_wait = bool(step_results and step_results[-1].decision.action.action.value == "wait")
        success = executed_steps > 0 or final_wait
        return ExecutionResult(
            success=success,
            method="vision_fallback",
            output=f"Vision loop finished with {executed_steps} executed step(s), budget={budget}.",
            error=None if success else "No executable step found.",
            duration_ms=0.0,
        )

    def ollama_is_available(self) -> bool:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        if base_url.endswith("/v1"):
            tags_url = base_url[:-3] + "/api/tags"
        else:
            tags_url = base_url.rstrip("/") + "/api/tags"

        try:
            req = request.Request(tags_url, method="GET")
            with request.urlopen(req, timeout=1) as resp:
                return 200 <= int(resp.status) < 300
        except Exception:
            return False

    @staticmethod
    def _safe_workspace_path(raw_path: str) -> Path | None:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate

        try:
            resolved = candidate.resolve()
            root = Path.cwd().resolve()
            resolved.relative_to(root)
            return resolved
        except Exception:
            return None

    @staticmethod
    def _extract_url(command: str) -> str | None:
        match = re.search(r"(https?://[^\s]+)", command, flags=re.IGNORECASE)
        if match:
            return match.group(1)

        web_hint = re.search(r"open\s+([a-z0-9.-]+\.[a-z]{2,})", command, flags=re.IGNORECASE)
        if web_hint:
            host = web_hint.group(1)
            if not host.startswith("http"):
                return f"https://{host}"
        return None

    @staticmethod
    def _open_known_app(target: str) -> tuple[bool, str, str | None] | None:
        app_map: dict[str, list[str]] = {
            "notepad": ["notepad.exe"],
            "calculator": ["calc.exe"],
            "calc": ["calc.exe"],
            "file explorer": ["explorer.exe"],
            "explorer": ["explorer.exe"],
            "vscode": ["code"],
            "visual studio code": ["code"],
            "chrome": ["chrome.exe"],
            "firefox": ["firefox.exe"],
        }

        normalized = target.strip().lower()
        website_map = {
            "youtube": "https://www.youtube.com",
            "gmail": "https://mail.google.com",
            "github": "https://github.com",
            "reddit": "https://www.reddit.com",
            "twitter": "https://twitter.com",
            "instagram": "https://www.instagram.com",
            "whatsapp": "https://web.whatsapp.com",
            "netflix": "https://www.netflix.com",
            "amazon": "https://www.amazon.in",
            "flipkart": "https://www.flipkart.com",
            "linkedin": "https://www.linkedin.com",
            "stackoverflow": "https://stackoverflow.com",
            "wikipedia": "https://en.wikipedia.org",
            "maps": "https://maps.google.com",
            "drive": "https://drive.google.com",
            "meet": "https://meet.google.com",
            "notion": "https://notion.so",
            "figma": "https://figma.com",
            "vercel": "https://vercel.com",
        }
        if normalized in website_map:
            webbrowser.open(website_map[normalized], new=2)
            return True, f"Opened {normalized} in browser.", None

        if normalized in {"settings", "windows settings"}:
            try:
                os.startfile("ms-settings:")  # type: ignore[attr-defined]
                return True, "Opened Windows Settings.", None
            except Exception as exc:
                return False, "", f"Could not open settings: {exc}"

        command = app_map.get(normalized)
        if command is None:
            return None

        try:
            subprocess.Popen(command)
            return True, f"Opened {normalized}.", None
        except FileNotFoundError:
            return False, "", f"App not found on this system: {normalized}"
        except Exception as exc:
            return False, "", f"Failed to open {normalized}: {exc}"

    @staticmethod
    def _extract_code_block(text: str) -> str:
        block = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if block:
            return block.group(1).strip()
        return text.strip()

    @staticmethod
    def _dangerous_code_reason(code: str) -> str | None:
        lowered = code.lower()
        forbidden = [
            "os.remove",
            "os.rmdir",
            "shutil.rmtree",
            "powershell",
            "cmd /c del",
            "format c:",
            "reg delete",
            "taskkill",
            "socket",
            "requests.delete",
        ]
        for token in forbidden:
            if token in lowered:
                return token

        if "subprocess" in lowered:
            destructive_subprocess = [
                "del ",
                "rmdir",
                "format",
                "rd /s",
                "rm -rf",
                "taskkill",
                "shutdown",
                "reg delete",
                "net user",
            ]
            if any(marker in lowered for marker in destructive_subprocess):
                return "Destructive subprocess command detected"

        if "open(" in lowered and "'w'" in lowered and "path(" not in lowered:
            return "Unscoped write operation"

        return None
