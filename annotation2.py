import tkinter as tk
import pyscreenshot as ImageGrab
from PIL import Image, ImageTk

class SnippingToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Snipping Tool")

        self.snip_button = tk.Button(root, text="Capture Snip", command=self.capture_snip)
        self.snip_button.pack(pady=10)

        self.canvas = tk.Canvas(root)
        self.canvas.pack()

        self.start_x = self.start_y = self.end_x = self.end_y = None

    def capture_snip(self):
        self.root.withdraw()  # Hide the main window temporarily
        snip = ImageGrab.grab(bbox=(0, 0, 1920, 1080))  # Capture the screen

        # Create a new window to display the snip
        snip_window = tk.Toplevel()
        snip_window.title("Snip")

        snip_tk = ImageTk.PhotoImage(snip)
        snip_label = tk.Label(snip_window, image=snip_tk)
        snip_label.pack()

        snip_window.protocol("WM_DELETE_WINDOW", lambda: self.on_snip_close(snip_window))

    def on_snip_close(self, snip_window):
        self.root.deiconify()  # Show the main window again
        snip_window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SnippingToolApp(root)
    root.mainloop()
