import os
import time
import subprocess

def open_application_cmd(app_name: str) -> bool:
    """
    Attempts to open an application directly using the command line or PATH.
    This works well for tools added to your system PATH (e.g., 'code' for VS Code, 'notepad', 'calc').
    
    Args:
        app_name (str): The executable name.
        
    Returns:
        bool: True if the command ran successfully, False otherwise.
    """
    try:
        # Popen is non-blocking, so the robot won't get stuck waiting for the app to close
        subprocess.Popen(app_name, shell=True)
        return True
    except Exception as e:
        print(f"Error opening '{app_name}' via command line: {e}")
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
        
        print(f"Opening '{app_name}' via Windows search...")
        # Press the Windows key
        pyautogui.press('win')
        time.sleep(1.0)  # Wait for the Start menu to open
        
        # Type the name of the application
        pyautogui.write(app_name, interval=0.05)
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

def open_app(app_name: str, use_gui: bool = True) -> bool:
    """
    General function to open an application.
    Defaults to the GUI method (Windows Key -> Search -> Enter) because it finds almost everything, 
    but can fall back to CMD if specified.
    """
    if use_gui:
        return open_application_gui(app_name)
    else:
        return open_application_cmd(app_name)

if __name__ == "__main__":
    # Example test
    # Trying to open VS Code via CMD (assuming 'code' is in PATH)
    print("Trying to open VS Code...")
    open_application_cmd("code")
    
    # Trying to open WhatsApp via Windows Search GUI simulation
    # (Uncomment below to test)
    # time.sleep(2) # brief pause before starting GUI automation
    # print("Trying to open WhatsApp...")
    # open_application_gui("WhatsApp")
