import os
import time
import shutil
import json
import subprocess

try:
    import winreg
except ImportError:
    winreg = None

KNOWN_ALIASES = {
    # Microsoft Office
    "word": "winword",
    "ms word": "winword",
    "microsoft word": "winword",
    "excel": "excel",
    "ms excel": "excel",
    "microsoft excel": "excel",
    "powerpoint": "powerpnt",
    "ppt": "powerpnt",
    "ms powerpoint": "powerpnt",
    "microsoft powerpoint": "powerpnt",
    "access": "msaccess",
    "outlook": "outlook",
    "onenote": "onenote",

    # Browsers
    "chrome": "chrome",
    "google chrome": "chrome",
    "edge": "msedge",
    "ms edge": "msedge",
    "microsoft edge": "msedge",
    "firefox": "firefox",
    "brave": "brave",

    # Utilities & Windows Tools
    "calc": "calc",
    "calculator": "calc",
    "notepad": "notepad",
    "paint": "mspaint",
    "mspaint": "mspaint",
    "snipping tool": "snippingtool",
    "snip": "snippingtool",
    "snippingtool": "snippingtool",
    "task manager": "taskmgr",
    "taskmgr": "taskmgr",
    "control panel": "control",
    "control": "control",
    "explorer": "explorer",
    "file explorer": "explorer",
    "files": "explorer",
    "settings": "ms-settings:",
    "windows settings": "ms-settings:",

    # Developer tools & Terminals
    "code": "code",
    "vscode": "code",
    "vs code": "code",
    "visual studio code": "code",
    "terminal": "wt",
    "windows terminal": "wt",
    "wt": "wt",
    "cmd": "cmd",
    "command prompt": "cmd",
    "powershell": "powershell",
    "powershell ise": "powershell_ise",

    # Popular apps
    "spotify": "spotify",
    "whatsapp": "whatsapp:",
    "teams": "msteams",
    "microsoft teams": "msteams",
    "zoom": "zoom",
    "vlc": "vlc",
}

_START_APPS_CACHE = None

def _get_start_apps() -> list[dict]:
    """Retrieves and caches the list of installed Windows applications."""
    global _START_APPS_CACHE
    if _START_APPS_CACHE is not None:
        return _START_APPS_CACHE
    try:
        cmd = ["powershell", "-NoProfile", "-Command", "Get-StartApps | Select-Object Name, AppID | ConvertTo-Json"]
        out = subprocess.check_output(cmd, text=True, errors="ignore", timeout=4)
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        _START_APPS_CACHE = data
    except Exception:
        _START_APPS_CACHE = []
    return _START_APPS_CACHE

def resolve_application(app_name: str) -> tuple[str, str] | None:
    """
    Resolves an application name to a runnable target.
    Returns (target_type, target_value) or None.
    target_type can be: 'path', 'protocol', 'start_app', or 'command'.
    """
    if not app_name:
        return None

    clean = app_name.strip().strip('"\'')
    # Strip common filler prefixes and suffixes
    for prefix in ["open ", "launch ", "start ", "run "]:
        if clean.lower().startswith(prefix):
            clean = clean[len(prefix):].strip()
    if clean.lower().endswith(" app"):
        clean = clean[:-4].strip()

    lower = clean.lower()
    target = KNOWN_ALIASES.get(lower, clean)

    # 1. Check if it's a URI protocol (e.g. ms-settings:, whatsapp:)
    if target.endswith(":") or (":" in target and not os.path.exists(target) and len(target.split(":")[0]) > 1):
        return ("protocol", target)

    # 2. Check if it's already an existing full file path
    if os.path.exists(target):
        return ("path", os.path.abspath(target))

    # 3. Check system PATH via shutil.which
    for cand in [target, f"{target}.exe", clean, f"{clean}.exe"]:
        found = shutil.which(cand)
        if found:
            return ("path", found)

    # 4. Check Windows Registry App Paths (HKLM & HKCU)
    if winreg:
        roots = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths"),
        ]
        # Exact / alias match in registry
        for cand in [target, f"{target}.exe", clean, f"{clean}.exe"]:
            key_cand = cand if cand.lower().endswith(".exe") else f"{cand}.exe"
            for root, subkey in roots:
                try:
                    with winreg.OpenKey(root, f"{subkey}\\{key_cand}") as k:
                        val = winreg.QueryValue(k, "")
                        if val:
                            clean_val = val.strip('"')
                            if os.path.exists(clean_val):
                                return ("path", clean_val)
                except Exception:
                    pass

        # Substring / fuzzy match across registered apps
        for root, subkey in roots:
            try:
                with winreg.OpenKey(root, subkey) as k:
                    num_keys = winreg.QueryInfoKey(k)[0]
                    for i in range(num_keys):
                        key_name = winreg.EnumKey(k, i)
                        base = key_name.lower().replace(".exe", "")
                        if base == lower or base == target.lower() or lower in base or base in lower:
                            with winreg.OpenKey(k, key_name) as app_k:
                                val = winreg.QueryValue(app_k, "")
                                if val:
                                    clean_val = val.strip('"')
                                    if os.path.exists(clean_val):
                                        return ("path", clean_val)
            except Exception:
                pass

    # 5. Check Windows StartApps (UWP & modern store apps)
    apps = _get_start_apps()
    for app in apps:
        name = app.get("Name", "").lower()
        if name == lower or name == target.lower():
            return ("start_app", f"shell:AppsFolder\\{app.get('AppID')}")

    for app in apps:
        name = app.get("Name", "").lower()
        if lower in name or target.lower() in name:
            return ("start_app", f"shell:AppsFolder\\{app.get('AppID')}")

    # 6. Fallback command name if single word
    if " " not in target:
        return ("command", target)

    return None

def open_application_cmd(app_name: str) -> bool:
    """
    Opens an application directly using its name via command line, system PATH,
    Windows Registry (App Paths), or Windows StartApps.
    Resolves natural application names (e.g., 'Word', 'Excel', 'PowerPoint', 'Chrome', 'calc')
    to their executable paths or system AppIDs, launching them non-blockingly.
    Falls back to Windows Search GUI simulation if direct resolution cannot launch.
    
    Args:
        app_name (str): The application name or executable name.
        
    Returns:
        bool: True if launched successfully, False otherwise.
    """
    try:
        resolved = resolve_application(app_name)
        if resolved:
            target_type, target_val = resolved
            print(f"[open_application_cmd] Resolved '{app_name}' -> ({target_type}) '{target_val}'")

            if target_type in ("protocol", "start_app"):
                if hasattr(os, "startfile"):
                    os.startfile(target_val)
                    return True
                else:
                    subprocess.Popen(f'explorer.exe "{target_val}"', shell=True)
                    return True

            elif target_type == "path":
                # Launch via os.startfile or start command
                if hasattr(os, "startfile"):
                    try:
                        os.startfile(target_val)
                        return True
                    except Exception as sf_err:
                        print(f"[open_application_cmd] os.startfile notice: {sf_err}, falling back to start command...")

                # Non-blocking start command
                subprocess.Popen(f'start "" "{target_val}"', shell=True)
                return True

            elif target_type == "command":
                # Check with shutil.which before attempting command
                exe = shutil.which(target_val)
                if exe:
                    subprocess.Popen(f'start "" "{exe}"', shell=True)
                    return True
                # If command exists in system registry, start command handles it
                subprocess.Popen(f'start "" "{target_val}"', shell=True)
                return True

        # If resolution didn't find a direct CLI target, fall back to GUI Windows Search
        print(f"[open_application_cmd] No direct CLI match for '{app_name}'. Falling back to Windows Search GUI...")
        return open_application_gui(app_name)

    except Exception as e:
        print(f"Error opening '{app_name}' via command line: {e}")
        try:
            return open_application_gui(app_name)
        except Exception:
            return False

def open_application_gui(app_name: str) -> bool:
    """
    Simulates a user pressing the Windows key, typing the application name,
    and pressing Enter to launch it. Very effective for UWP apps like WhatsApp.
    
    Args:
        app_name (str): The name of the application to search for.
        
    Returns:
        bool: True if execution succeeded, False otherwise.
    """
    try:
        import pyautogui
        
        # Clean app name if needed
        clean = app_name.strip().strip('"\'')
        for prefix in ["open ", "launch ", "start ", "run "]:
            if clean.lower().startswith(prefix):
                clean = clean[len(prefix):].strip()
        if clean.lower().endswith(" app"):
            clean = clean[:-4].strip()

        print(f"Opening '{clean}' via Windows search...")
        # Press the Windows key
        pyautogui.press('win')
        time.sleep(1.0)  # Wait for the Start menu to open
        
        # Type the name of the application
        pyautogui.write(clean, interval=0.05)
        time.sleep(1.5)  # Wait for Windows Search to find the best match
        
        # Press enter to open it
        pyautogui.press('enter')
        return True
    except ImportError:
        print("Error: The 'pyautogui' module is not installed. Run 'pip install pyautogui'.")
        return False
    except Exception as e:
        print(f"Error opening '{app_name}' via GUI: {e}")
        return False

def open_app(app_name: str, use_gui: bool = False) -> bool:
    """
    General function to open an application.
    Tries fast direct launch via CMD/Registry/StartApps resolver first.
    Falls back to Windows Search GUI if specified or if direct launch fails.
    """
    if use_gui:
        return open_application_gui(app_name)
    success = open_application_cmd(app_name)
    if not success:
        return open_application_gui(app_name)
    return True

if __name__ == "__main__":
    print("Testing open_application_cmd with 'Word'...")
    res = open_application_cmd("Word")
    print("Result:", res)
