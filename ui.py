import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import cv2
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def make_heatmap(mods, out_path="heatmap.png"):
    if not mods or len(mods) == 0:
        messagebox.showinfo("Info", "No module data to visualize.")
        return None
    w, h = 800, 600
    base_img = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    mods = np.array(mods, dtype=float)
    if mods.shape[1] < 4:
        raise ValueError("Each module must have x, y, w, h values")
    x, y, ww, hh = mods[:, 0], mods[:, 1], mods[:, 2], mods[:, 3]
    cx, cy = x + ww / 2, y + hh / 2
    weights = ww * hh
    H, _, _ = np.histogram2d(cx, cy, bins=[80, 60],
                             range=[[0, w], [0, h]],
                             weights=weights)
    H = H.T
    H = gaussian_filter(H, sigma=2)
    if np.max(H) > 0:
        H = H / np.max(H)
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.axis("off")
    ax.imshow(base_img, extent=(0, w, h, 0))
    ax.imshow(H, cmap="jet", alpha=0.6, extent=(0, w, h, 0), origin="upper")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path


def detect_colors():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Camera not found or busy")
        return
    ranges = {
        "Red": ([0, 100, 100], [10, 255, 255]),
        "Green": ([40, 50, 50], [90, 255, 255]),
        "Blue": ([100, 150, 0], [140, 255, 255])
    }
    draw_col = {"Red": (0, 0, 255), "Green": (0, 255, 0), "Blue": (255, 0, 0)}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for name, (low, up) in ranges.items():
            low_np, up_np = np.array(low, dtype=np.uint8), np.array(up, dtype=np.uint8)
            mask = cv2.inRange(hsv, low_np, up_np)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) > 500:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), draw_col[name], 2)
                    cv2.putText(frame, name, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_col[name], 2)
        cv2.imshow("Color Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


IMG_SIZE = (128, 128)
MODEL_PATH = "module_classifier.h5"


def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            messagebox.showerror("Model Error", f"Cannot load model:\n{e}")
            return None
    else:
        messagebox.showinfo("Info", "Model file not found.")
        return None


def classify(img_path):
    model = load_model()
    if model is None:
        return
    try:
        img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
        arr = tf.keras.utils.img_to_array(img)
        arr = np.expand_dims(arr, 0)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        pred = model.predict(arr)
        cls = int(np.argmax(pred[0]))
        names = ["life_support", "power", "sleep", "exercise"]
        result = f"{names[cls]} ({pred[0][cls]:.2f})"
        messagebox.showinfo("Classification Result", result)
    except Exception as e:
        messagebox.showerror("Error", f"Classification failed:\n{e}")


root = tk.Tk()
root.title("HabitatLab")
root.geometry("800x600")
c = tk.Canvas(root, bg="white", width=600, height=400)
c.pack(pady=10)


def start_draw(e):
    c.start_x, c.start_y = e.x, e.y
    c.rect = c.create_rectangle(e.x, e.y, e.x, e.y, outline="black")


def draw_rect(e):
    c.coords(c.rect, c.start_x, c.start_y, e.x, e.y)


c.bind("<Button-1>", start_draw)
c.bind("<B1-Motion>", draw_rect)
area = tk.StringVar(value="Area: 0")
mods = tk.StringVar(value="Modules: 0")
stat = tk.StringVar(value="Status: idle")
f = tk.Frame(root)
f.pack(pady=10)
tk.Label(f, textvariable=area).grid(row=0, column=0, padx=10)
tk.Label(f, textvariable=mods).grid(row=0, column=1, padx=10)
tk.Label(f, textvariable=stat).grid(row=0, column=2, padx=10)


def start_colors():
    threading.Thread(target=detect_colors, daemon=True).start()


def classify_file():
    path = filedialog.askopenfilename(title="Select image",
                                      filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if path:
        classify(path)


def heatmap_btn():
    mods_sample = [(10, 20, 50, 40), (100, 150, 60, 80), (400, 300, 120, 90)]
    out = make_heatmap(mods_sample, "hm_gui.png")
    if out:
        messagebox.showinfo("Heatmap", f"Saved to {out}")


tk.Button(root, text="Detect Colors", command=start_colors,
          bg="lightgreen", width=20).pack(pady=5)
tk.Button(root, text="Classify Image", command=classify_file,
          bg="lightblue", width=20).pack(pady=5)
tk.Button(root, text="Generate Heatmap", command=heatmap_btn,
          bg="orange", width=20).pack(pady=5)
root.mainloop()
