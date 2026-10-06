import pyautogui
import time
from typing import Tuple, Optional

# Failsafe: if the robot goes out of control, moving the mouse to any of the 4 corners of the screen will abort the program.
pyautogui.FAILSAFE = True

def type_text(text: str, interval: float = 0.05) -> bool:
    """
    Types out the given text character by character, acting like a real user.
    
    Args:
        text (str): The text to type.
        interval (float): The delay between key presses.
    """
    try:
        # Brief pause to ensure the target window is focused before typing starts
        time.sleep(1.5)
        pyautogui.write(text, interval=interval)
        return True
    except Exception as e:
        print(f"Error typing text: {e}")
        return False

def press_key(key: str) -> bool:
    """
    Presses a single keyboard key.
    Examples of keys: 'enter', 'tab', 'esc', 'space', 'backspace', 'up', 'down', 'win'.
    """
    try:
        pyautogui.press(key)
        return True
    except Exception as e:
        print(f"Error pressing key '{key}': {e}")
        return False

def press_hotkey(keys: list) -> bool:
    """
    Presses a combination of keys together.
    Example usage: press_hotkey(keys=['ctrl', 'c']) or press_hotkey(keys=['alt', 'tab'])
    """
    try:
        pyautogui.hotkey(*keys)
        return True
    except Exception as e:
        print(f"Error pressing hotkey {keys}: {e}")
        return False

def move_mouse(x: int, y: int, duration: float = 0.2) -> bool:
    """
    Moves the mouse pointer to the absolute (x, y) coordinates on the screen.
    
    Args:
        x (int): The x coordinate.
        y (int): The y coordinate.
        duration (float): How long the movement should take in seconds (makes it look human).
    """
    try:
        pyautogui.moveTo(x, y, duration=duration)
        return True
    except Exception as e:
        print(f"Error moving mouse to ({x}, {y}): {e}")
        return False

def click(x: Optional[int] = None, y: Optional[int] = None, button: str = 'left', clicks: int = 1) -> bool:
    """
    Clicks the mouse. 
    If x and y are provided, it moves the mouse to those coordinates first.
    If x and y are None, it clicks at the current mouse position.
    
    Args:
        x (int, optional): The x coordinate.
        y (int, optional): The y coordinate.
        button (str): 'left', 'right', or 'middle'.
        clicks (int): Number of clicks.
    """
    try:
        pyautogui.click(x=x, y=y, button=button, clicks=clicks)
        return True
    except Exception as e:
        print(f"Error clicking at ({x}, {y}) with {button} button: {e}")
        return False

def double_click(x: Optional[int] = None, y: Optional[int] = None) -> bool:
    """
    Double clicks the left mouse button.
    """
    try:
        pyautogui.doubleClick(x=x, y=y)
        return True
    except Exception as e:
        print(f"Error double clicking at ({x}, {y}): {e}")
        return False

def right_click(x: Optional[int] = None, y: Optional[int] = None) -> bool:
    """
    Right clicks the mouse.
    """
    try:
        pyautogui.rightClick(x=x, y=y)
        return True
    except Exception as e:
        print(f"Error right clicking at ({x}, {y}): {e}")
        return False

def get_mouse_position() -> Tuple[int, int]:
    """
    Returns the current (x, y) coordinates of the mouse.
    Can be used to record locations for later clicking.
    """
    try:
        return pyautogui.position()
    except Exception as e:
        print(f"Error getting mouse position: {e}")
        return (0, 0)

def take_screenshot() -> bool:
    """
    Takes a screenshot by pressing the printscreen key, and then dragging the mouse
    from the top-left to the bottom-right of the screen to capture the area.
    """
    try:
        # Press print screen to open snipping tool / screenshot UI
        pyautogui.press('printscreen')
        time.sleep(3) # Wait for the UI to appear
        
        # Get screen dimensions
        width, height = pyautogui.size()
        
        # Move to top left corner (use 5, 5 to safely avoid PyAutoGUI failsafe at 0,0)
        pyautogui.moveTo(5, 5)
        
        # Drag to bottom right corner (subtract 5 to safely avoid bottom-right failsafe)
        pyautogui.dragTo(width - 5, height - 5, duration=0.5, button='left')
        
        # Click to finalize the selection (just in case the drag release isn't registered)
        time.sleep(3)
        pyautogui.click()
        return True
    except Exception as e:
        print(f"Error taking screenshot: {e}")
        return False

if __name__ == "__main__":
    # Example usage / testing
    print("Mouse position is:", get_mouse_position())
    
    # print("Testing typing...")
    # type_text("Hello, I am a robot!", interval=0.01)
    
    # print("Testing hotkey (Selecting all text)...")
    # press_hotkey('ctrl', 'a')
