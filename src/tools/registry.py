"""
Tools Registry
This file maintains a hierarchical structure (tree) of all available tools.
It is designed to be fed to an SLM (Small Language Model) in stages to save context window space.
"""
from typing import Dict

TOOLS_REGISTRY = {
    "browser": {
        "description": "Tools to interact with the internet. Use this to open specific URLs, visit websites, or search for information using search engines.",
        "tools": {
            "open_url": "Opens an exact URL string (e.g., 'https://github.com') in the default web browser.",
            "search_browser": "Searches for a text query in a browser using engines like google, bing, or duckduckgo.",
            "get_unread_emails": "Fetches the latest unread emails from the user's Gmail inbox.",
            "send_email": "Sends an email using the user's Gmail account.",
            "wait_for_new_email": "Continuously polls the Gmail API until a new email arrives in the inbox.",
            "get_upcoming_events": "Fetches upcoming events from the user's Google Calendar.",
            "create_event": "Creates an event in the user's primary Google Calendar."
        }
    },
    "application": {
        "description": "Tools to launch and open local desktop applications installed on the laptop (e.g., WhatsApp, VS Code, Notepad).",
        "tools": {
            "open_app": "The primary tool to open any application. It finds the app by name and launches it.",
            "open_application_gui": "Simulates pressing the Windows key, typing the app name, and pressing Enter to open an app.",
            "open_application_cmd": "Fast-tracks opening system apps directly via command line (only for apps already in system PATH)."
        }
    },
    "interaction": {
        "description": "Tools to simulate physical human input (keyboard and mouse). Use this to click on screen coordinates, type text, or press key combinations.",
        "tools": {
            "type_text": "Types out a given string of text character by character.",
            "press_key": "Presses a single keyboard key (e.g., 'enter', 'tab', 'esc', 'space').",
            "press_hotkey": "Presses multiple keys simultaneously (e.g., 'ctrl', 'c' or 'alt', 'tab').",
            "move_mouse": "Moves the mouse cursor to absolute (x, y) screen pixel coordinates.",
            "click": "Clicks the mouse. Can specify (x,y) coordinates and button type (left, right, middle).",
            "double_click": "Double-clicks the left mouse button.",
            "right_click": "Right-clicks the mouse.",
            "get_mouse_position": "Gets current (x, y) coordinates of the mouse cursor. Useful for recording where to click.",
            "take_screenshot": "Takes a screenshot of the entire screen by pressing printscreen and clicking and dragging the mouse."
        }
    },
    "file_explorer": {
        "description": "Tools to manage the local file system. Use this to search, move, copy, rename, and list files and directories.",
        "tools": {
            "search_files": "Searches for files or folders by name recursively within a directory.",
            "move_file": "Moves a file or directory from source to destination (acts as cut and move).",
            "copy_file": "Copies a file or directory from source to destination.",
            "rename_file": "Renames a file or directory.",
            "list_directory": "Lists all files and folders in a directory.",
            "get_all_file_names": "Returns a list of all files (excluding directories) in the specified folder.",
            "create_folder": "Creates a new folder inside the specified directory.",
            "get_file_metadata": "Gets metadata for a file, including size (bytes) and modification time.",
            "delete_item": "Deletes a file or an empty directory."
        }
    }
}

def get_top_level_categories() -> Dict[str, str]:
    """
    Returns only the top-level categories and their descriptions.
    Pass this to the SLM FIRST. It allows the model to choose the general domain 
    (browser vs app vs interaction) without being overwhelmed by every single tool.
    """
    return {
        category: data["description"] 
        for category, data in TOOLS_REGISTRY.items()
    }

def get_category_tools(category: str) -> Dict[str, str]:
    """
    Returns the detailed dictionary of specific tools inside a category.
    Pass this to the SLM SECOND, after it has chosen a category, so it can pick the exact tool.
    """
    if category in TOOLS_REGISTRY:
        return TOOLS_REGISTRY[category]["tools"]
    return {"error": f"Category '{category}' not found."}

def get_slm_tool_schemas() -> list:
    """
    Returns the complete OpenAI-compatible JSON schemas for all tools in the registry.
    This is used by SLMs like Hammer to understand exact function signatures.
    """
    return [
        # BROWSER TOOLS
        {"type": "function", "function": {"name": "open_url", "description": "Opens an exact URL string.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
        {"type": "function", "function": {"name": "search_browser", "description": "Searches for a text query in a browser.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "get_unread_emails", "description": "Fetches the latest unread emails from Gmail.", "parameters": {"type": "object", "properties": {"max_results": {"type": "integer"}}, "required": []}}},
        {"type": "function", "function": {"name": "send_email", "description": "Sends an email via Gmail.", "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "subject", "body"]}}},
        {"type": "function", "function": {"name": "wait_for_new_email", "description": "Polls Gmail until a new email arrives.", "parameters": {"type": "object", "properties": {"poll_interval_seconds": {"type": "integer"}}, "required": []}}},
        {"type": "function", "function": {"name": "get_upcoming_events", "description": "Fetches upcoming Google Calendar events.", "parameters": {"type": "object", "properties": {"max_results": {"type": "integer"}}, "required": []}}},
        {"type": "function", "function": {"name": "create_event", "description": "Creates an event in Google Calendar.", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}, "description": {"type": "string"}, "start_time_iso": {"type": "string"}, "end_time_iso": {"type": "string"}}, "required": ["summary", "description", "start_time_iso", "end_time_iso"]}}},
        
        # APPLICATION TOOLS
        {"type": "function", "function": {"name": "open_app", "description": "The primary tool to open any application.", "parameters": {"type": "object", "properties": {"app_name": {"type": "string"}}, "required": ["app_name"]}}},
        {"type": "function", "function": {"name": "open_application_gui", "description": "Simulates pressing Windows key and typing to open an app.", "parameters": {"type": "object", "properties": {"app_name": {"type": "string"}}, "required": ["app_name"]}}},
        {"type": "function", "function": {"name": "open_application_cmd", "description": "Fast-tracks opening system apps directly via command line.", "parameters": {"type": "object", "properties": {"app_name": {"type": "string"}}, "required": ["app_name"]}}},
        
        # INTERACTION TOOLS
        {"type": "function", "function": {"name": "type_text", "description": "Types out a given string of text character by character.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
        {"type": "function", "function": {"name": "press_key", "description": "Presses a single keyboard key (e.g., 'enter', 'esc').", "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}},
        {"type": "function", "function": {"name": "press_hotkey", "description": "Presses multiple keys simultaneously.", "parameters": {"type": "object", "properties": {"hotkey": {"type": "string"}}, "required": ["hotkey"]}}},
        {"type": "function", "function": {"name": "move_mouse", "description": "Moves the mouse cursor to absolute (x, y) coordinates.", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]}}},
        {"type": "function", "function": {"name": "click", "description": "Clicks the mouse.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "double_click", "description": "Double-clicks the left mouse button.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "right_click", "description": "Right-clicks the mouse.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "get_mouse_position", "description": "Gets current (x, y) coordinates of the mouse cursor.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "take_screenshot", "description": "Takes a screenshot of the screen.", "parameters": {"type": "object", "properties": {}}}},
        
        # SYSTEM TOOLS
        {"type": "function", "function": {"name": "goal_complete", "description": "Call this when all steps to satisfy the original goal have been completed.", "parameters": {"type": "object", "properties": {}}}},
        
        # FILE EXPLORER TOOLS
        {"type": "function", "function": {"name": "search_files", "description": "Searches for files or folders by name recursively within a directory.", "parameters": {"type": "object", "properties": {"directory": {"type": "string"}, "name": {"type": "string"}}, "required": ["directory", "name"]}}},
        {"type": "function", "function": {"name": "move_file", "description": "Moves a file or directory from source to destination.", "parameters": {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]}}},
        {"type": "function", "function": {"name": "copy_file", "description": "Copies a file or directory from source to destination.", "parameters": {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]}}},
        {"type": "function", "function": {"name": "rename_file", "description": "Renames a file or directory.", "parameters": {"type": "object", "properties": {"old_path": {"type": "string"}, "new_path": {"type": "string"}}, "required": ["old_path", "new_path"]}}},
        {"type": "function", "function": {"name": "list_directory", "description": "Lists all files and folders in a directory.", "parameters": {"type": "object", "properties": {"directory": {"type": "string"}}, "required": ["directory"]}}},
        {"type": "function", "function": {"name": "get_all_file_names", "description": "Returns a list of all files (excluding directories) in the specified folder.", "parameters": {"type": "object", "properties": {"directory": {"type": "string"}}, "required": ["directory"]}}},
        {"type": "function", "function": {"name": "create_folder", "description": "Creates a new folder inside the specified directory.", "parameters": {"type": "object", "properties": {"directory": {"type": "string"}, "folder_name": {"type": "string"}}, "required": ["directory", "folder_name"]}}},
        {"type": "function", "function": {"name": "get_file_metadata", "description": "Gets metadata for a file, including size (bytes) and modification time.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "delete_item", "description": "Deletes a file or an empty directory.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
    ]
