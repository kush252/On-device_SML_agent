import os
import re
import json
import shutil
import subprocess
from typing import List

def open_folder(directory: str) -> bool:
    """
    Opens a folder directly in Windows File Explorer.
    
    Args:
        directory (str): Path of the folder to open.
        
    Returns:
        bool: True if folder was opened, False otherwise.
    """
    try:
        clean_dir = directory.strip().strip('"\'')
        if not os.path.exists(clean_dir):
            print(f"[open_folder] Directory '{clean_dir}' does not exist.")
            return False
            
        print(f"[open_folder] Opening folder in File Explorer: '{clean_dir}'")
        if hasattr(os, "startfile"):
            os.startfile(clean_dir)
        else:
            subprocess.Popen(f'explorer.exe "{clean_dir}"', shell=True)
        return True
    except Exception as e:
        print(f"[open_folder] Error opening folder '{directory}': {e}")
        return False

def _sanitize_folder_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', name).strip().strip('.')
    return cleaned or "Uncategorized"

def _group_files_by_name(files: List[str]) -> dict:
    """
    Groups a list of filenames into logical topic/category names.
    Uses semantic keyword scoring with local SLM / Ollama fallback.
    """
    TOPIC_KEYWORDS = {
        "AI_and_Code": [
            "llm", "large language model", "code", "coding", "vulnerability",
            "repair", "patch", "software", "generation", "prompt", "neural",
            "transformer", "token"
        ],
        "Medical": [
            "medical", "health", "clinical", "imaging", "doctor", "disease",
            "patient", "biomedical", "multimodal medical", "radiology"
        ],
        "Finance": [
            "finance", "financial", "bank", "tokenbank", "market", "trading",
            "investment", "money", "economy", "crypto", "decision making"
        ],
        "Computer_Vision": [
            "vision", "image", "detection", "segmentation", "video", "visual"
        ],
        "Security_and_Privacy": [
            "security", "privacy", "attack", "defense", "cipher", "crypto"
        ]
    }

    category_map = {}
    assigned = set()

    # Step 1: Accurate keyword topic matching
    for f in files:
        name_lower = f.lower()
        matched_cat = None
        max_score = 0
        for cat, keywords in TOPIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in name_lower)
            if score > max_score:
                max_score = score
                matched_cat = cat

        if matched_cat and max_score > 0:
            category_map.setdefault(matched_cat, []).append(f)
            assigned.add(f)

    unassigned = [f for f in files if f not in assigned]

    # Step 2: For any unassigned files, try local Ollama model if available
    if unassigned:
        try:
            import ollama
            prompt = (
                "You are a file management assistant. Group these filenames into 2-4 clean category names.\n"
                "Respond ONLY with a JSON dictionary in format {\"CategoryName\": [\"filename1\", ...]} without code fences.\n\n"
                f"Files:\n{json.dumps(unassigned)}"
            )
            resp = ollama.chat(
                model="hf.co/keisuke-miyako/Hammer2.1-1.5b-gguf-q4_k_m:Q4_K_M",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0}
            )
            content = resp.get("message", {}).get("content", "").strip()
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                parsed = json.loads(content[start:end])
                for cat, f_list in parsed.items():
                    if isinstance(f_list, list):
                        clean_cat = _sanitize_folder_name(cat)
                        matched = [f for f in f_list if f in unassigned]
                        if matched:
                            category_map.setdefault(clean_cat, []).extend(matched)
                            assigned.update(matched)
        except Exception:
            pass

    # Step 3: For remaining files, fallback to extension or general
    remaining = [f for f in files if f not in assigned]
    for f in remaining:
        ext = os.path.splitext(f)[1].replace(".", "").upper()
        cat = f"{ext}_Files" if ext else "Other_Files"
        category_map.setdefault(cat, []).append(f)

    return category_map

def organize_files(directory: str) -> bool:
    """
    Organizes files in a directory into categorized subfolders by reading their names,
    clustering them into logical groups, creating subfolders, and moving the files.

    Args:
        directory (str): The path to the folder containing the files to organize.

    Returns:
        bool: True if files were successfully organized, False otherwise.
    """
    try:
        clean_dir = directory.strip().strip('"\'')
        if not os.path.exists(clean_dir):
            print(f"[organize_files] Error: Directory '{clean_dir}' does not exist.")
            return False

        if not os.path.isdir(clean_dir):
            print(f"[organize_files] Error: Path '{clean_dir}' is not a directory.")
            return False

        # Find all files in the directory (ignore subdirectories already present)
        files = [
            f for f in os.listdir(clean_dir)
            if os.path.isfile(os.path.join(clean_dir, f)) and not f.startswith(".") and f not in ("desktop.ini", "Thumbs.db")
        ]

        if not files:
            print(f"[organize_files] No loose files found to organize in '{clean_dir}'.")
            return True

        print(f"[organize_files] Found {len(files)} file(s) in '{clean_dir}'. Grouping by name/topic...")
        category_map = _group_files_by_name(files)

        moved_count = 0
        for category_name, cat_files in category_map.items():
            folder_path = os.path.join(clean_dir, category_name)
            os.makedirs(folder_path, exist_ok=True)

            for filename in cat_files:
                src_path = os.path.join(clean_dir, filename)
                dst_path = os.path.join(folder_path, filename)
                if os.path.exists(src_path):
                    shutil.move(src_path, dst_path)
                    moved_count += 1

        print(f"[organize_files] Successfully organized {moved_count} file(s) into {len(category_map)} folder(s):")
        for cat, flist in category_map.items():
            print(f"  [Folder] {cat} ({len(flist)} files):")
            for f in flist:
                print(f"     - {f}")

        return True
    except Exception as e:
        print(f"[organize_files] Error organizing files in '{directory}': {e}")
        return False

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
