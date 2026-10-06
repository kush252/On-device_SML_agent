import os
import shutil
from typing import List

def search_files(directory: str, name: str) -> List[str]:
    """Searches for files or folders by name recursively within a directory."""
    try:
        results = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if name.lower() in file.lower():
                    results.append(os.path.join(root, file))
            for d in dirs:
                if name.lower() in d.lower():
                    results.append(os.path.join(root, d))
        return results
    except Exception as e:
        print(f"Error searching for '{name}' in '{directory}': {e}")
        return []

def move_file(source: str, destination: str) -> bool:
    """Moves a file or directory from source to destination (acts as cut and move)."""
    try:
        shutil.move(source, destination)
        return True
    except Exception as e:
        print(f"Error moving '{source}' to '{destination}': {e}")
        return False

def copy_file(source: str, destination: str) -> bool:
    """Copies a file or directory from source to destination."""
    try:
        if os.path.isdir(source):
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        return True
    except Exception as e:
        print(f"Error copying '{source}' to '{destination}': {e}")
        return False

def rename_file(old_path: str, new_path: str) -> bool:
    """Renames a file or directory."""
    try:
        os.rename(old_path, new_path)
        return True
    except Exception as e:
        print(f"Error renaming '{old_path}' to '{new_path}': {e}")
        return False

def list_directory(directory: str) -> List[str]:
    """Lists all files and folders in a directory."""
    try:
        return os.listdir(directory)
    except Exception as e:
        print(f"Error listing directory '{directory}': {e}")
        return []

def get_all_file_names(directory: str) -> List[str]:
    """Returns a list of all files (excluding directories) in the specified folder."""
    try:
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    except Exception as e:
        print(f"Error getting files in '{directory}': {e}")
        return []

def create_folder(directory: str, folder_name: str) -> bool:
    """Creates a new folder inside the specified directory."""
    try:
        path = os.path.join(directory, folder_name)
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating folder '{folder_name}' in '{directory}': {e}")
        return False

def get_file_metadata(path: str) -> dict:
    """Gets metadata for a file, including size (bytes) and modification time."""
    try:
        if os.path.exists(path):
            stat = os.stat(path)
            return {
                "size_bytes": stat.st_size,
                "last_modified": stat.st_mtime,
                "is_dir": os.path.isdir(path)
            }
        return {}
    except Exception as e:
        print(f"Error getting metadata for '{path}': {e}")
        return {}

def delete_item(path: str) -> bool:
    """Deletes a file or an empty directory."""
    try:
        if os.path.isdir(path):
            os.rmdir(path)
        else:
            os.remove(path)
        return True
    except Exception as e:
        print(f"Error deleting '{path}': {e}")
        return False
