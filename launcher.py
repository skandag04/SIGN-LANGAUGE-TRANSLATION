import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os

# Get the base path to correctly find the other scripts
# This assumes ML_classifier.py, yolo_resnet.py, and words_classifier.py are in the same directory.
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# --- Core Logic Functions ---

def run_sign_detector(script_name):
    """Launches the specified Python script in a new process."""
    file_path = os.path.join(BASE_PATH, script_name)
    
    if not os.path.exists(file_path):
        messagebox.showerror("Error", f"Could not find {script_name}. Please ensure it is in the same directory.")
        return

    try:
        # Use subprocess.Popen to execute the script without blocking the UI
        subprocess.Popen([sys.executable, file_path])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to launch {script_name}: {e}")

def run_numbers():
    run_sign_detector("ML_classifier.py")

def run_alphabets():
    run_sign_detector("yolo_resnet.py")

def run_words():
    """Launches the words classifier script."""
    run_sign_detector("words_classifier.py")

# --- UI Transition Functions ---

def show_selection_screen():
    """Hides the start button and shows the alphabet/number/word selection buttons."""
    # Hide all initial elements
    start_button.pack_forget()
    title.pack_forget()
    
    # Repack the title at the top
    title.pack(pady=50)

    # Show selection elements
    subtitle.pack(pady=(0, 20))
    btn_alphabets.pack(pady=10)
    btn_numbers.pack(pady=10)
    btn_words.pack(pady=10) # Added the Words button to the packing list

# --- UI Setup ---
root = tk.Tk()
root.title("Sign Language Detector Launcher")

# Configure styles
primary_color = "#34A853"   # Google Green (Numbers)
secondary_color = "#4285F4" # Google Blue (Alphabets)
words_color = "#FBBC05"     # Google Yellow (Words) - NEW COLOR
start_color = "#EA4335"     # Google Red (Start button)
bg_color = "#F5F5DC"        # Cream White (Beige) Background
fg_color = "#333333"        # Dark Gray Foreground/Text

root.configure(bg=bg_color)

# MAKE IT FULL SCREEN
root.attributes('-fullscreen', True) 

# Ensure the window is closed properly when pressing 'Escape' or 'q'
def exit_fullscreen(event):
    root.destroy()
root.bind('<Escape>', exit_fullscreen)
root.bind('q', exit_fullscreen)


# Custom Button Style Function
def create_styled_button(parent, text, command, color, font_size=12):
    """Helper function to create styled buttons."""
    return tk.Button(parent, text=text,
                     command=command, 
                     width=30, 
                     height=3,
                     bg=color, 
                     fg="white", 
                     font=("Segoe UI", font_size, "bold"),
                     relief=tk.FLAT,
                     activebackground=color,
                     activeforeground="white",
                     bd=0, 
                     cursor="hand2")

# --- Shared UI Elements ---

# Title Label
title = tk.Label(root, text="Sign Language Detection",
                 fg=fg_color, bg=bg_color, font=("Segoe UI", 36, "bold"), justify=tk.CENTER)

# Subtitle
subtitle = tk.Label(root, text="Select a detection mode to begin:",
                    fg=fg_color, bg=bg_color, font=("Segoe UI", 16, "bold"))

# Button 1: Alphabets 
btn_alphabets = create_styled_button(root, "Detect Alphabets (A-Z)", run_alphabets, secondary_color)

# Button 2: Numbers
btn_numbers = create_styled_button(root, "Detect Numbers (0-9)", run_numbers, primary_color)

# Button 3: Words (NEW)
btn_words = create_styled_button(root, "Detect Words", run_words, words_color)

# --- Initial Start Screen Setup ---

# The START Button
start_button = create_styled_button(root, "START RECOGNITION", show_selection_screen, start_color, font_size=18)

# Initial placement for full screen
title.pack(pady=150) 
start_button.pack(pady=20)

# Start the Tkinter event loop
root.mainloop()