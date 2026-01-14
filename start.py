# start.py

import os
import sys
import time
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog

# ------------------------------
# Loading animation controller
# ------------------------------
_stop_loading = threading.Event()


def loading_animation(message: str):
    """Displays a loading spinner animation."""
    spinner = ['◜', '◝', '◞', '◟']
    i = 0
    while not _stop_loading.is_set():
        sys.stdout.write(f"\r{message} {spinner[i % len(spinner)]}")
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)


# ------------------------------
# Folder picker
# ------------------------------
def pick_folder() -> str:
    """Prompts the user to select a folder via GUI or terminal input."""
    try:
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="Select Folder to Create VENV")
        root.destroy()

        if not folder:
            print(">--Setup cancelled by user--<")
            sys.exit(0)

        return os.path.abspath(folder)

    except Exception:
        print("⚠ GUI failed. Falling back to terminal input.")
        folder = input("Enter full path for venv manually: ").strip()
        if not folder:
            print(">--No path provided. Exiting.--<")
            sys.exit(1)
        return os.path.abspath(folder)


# ------------------------------
# Virtual environment creation
# ------------------------------
def create_virtual_environment(venv_dir: str) -> None:
    """Creates a virtual environment at the specified directory."""
    if os.path.exists(venv_dir):
        print(f">--Virtual environment already exists at: {venv_dir}--<")
        return

    print(f">--Creating virtual environment at: {venv_dir}--<")

    _stop_loading.clear()
    spinner_thread = threading.Thread(
        target=loading_animation,
        args=("Creating virtual environment",),
        daemon=True
    )
    spinner_thread.start()

    try:
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to create virtual environment: {e}")
        sys.exit(1)
    finally:
        _stop_loading.set()
        spinner_thread.join()
        print("\r>--Virtual environment created successfully--<")


# ------------------------------
# Install requirements
# ------------------------------
def install_requirements(venv_dir: str) -> None:
    print(">--Installing required packages--<")

    project_root = os.path.dirname(os.path.abspath(__file__))
    installer_script = os.path.join(project_root, "utils", "install_req.py")

    if not os.path.exists(installer_script):
        print("❌ install_req.py not found.")
        sys.exit(1)

    # Resolve venv python explicitly
    if os.name == "nt":
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")

    if not os.path.exists(venv_python):
        print("❌ Venv python not found.")
        sys.exit(1)

    try:
        subprocess.check_call([
            venv_python,
            installer_script,
            venv_dir
        ])
    except subprocess.CalledProcessError as e:
        print(f"❌ Package installation failed: {e}")
        sys.exit(1)


# ------------------------------
# Post-setup instructions
# ------------------------------
def print_next_steps(venv_dir: str) -> None:
    print("\n>-- NEXT STEPS --<\n")

    if os.name == "nt":
        activate_cmd = os.path.join(venv_dir, "Scripts", "activate")
        python_cmd = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        activate_cmd = f"source {os.path.join(venv_dir, 'bin', 'activate')}"
        python_cmd = os.path.join(venv_dir, "bin", "python")

    print("1) Activate the virtual environment:")
    print(f"   {activate_cmd}\n")

    print("2) Run the Streamlit UI:")
    print("   streamlit run ui/streamlit_app.py\n")

    print("OR (without activation):")
    print(f"   {python_cmd} -m streamlit run ui/streamlit_app.py\n")


# ------------------------------
# Entry point
# ------------------------------
def main() -> None:
    print("\n>-- ADCP Setup Initializing --<\n")

    venv_base_path = pick_folder()
    venv_dir = os.path.join(venv_base_path, "ADCP")

    create_virtual_environment(venv_dir)
    install_requirements(venv_dir)

    print("\n>-- Setup completed successfully --<")
    print_next_steps(venv_dir)


if __name__ == "__main__":
    main()
