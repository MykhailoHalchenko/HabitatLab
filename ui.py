import tkinter as tk
from tkinter import messagebox
import main

def evaluate_habitat():
    main.main()
    messagebox.showinfo("Evaluate", "Запущено аналіз ML та Sionna.")

def update_metrics():
    area = '0'
    modules = '0'
    connection_status = "Запустіть аналіз"

    area_var.set(f"Площа: {area} м²")
    modules_var.set(f"Модулі: {modules}")
    status_var.set(f"Зв'язок: {connection_status}")

def start_draw(event):
    canvas.start_x, canvas.start_y = event.x, event.y
    canvas.current_rect = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="black", width=2)

def draw_rectangle(event):
    canvas.coords(canvas.current_rect, canvas.start_x, canvas.start_y, event.x, event.y)

root = tk.Tk()
root.title("HabitatLab")
root.geometry("800x600")

canvas = tk.Canvas(root, bg="white", width=600, height=400)
canvas.pack(pady=10)

canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw_rectangle)


metrics_frame = tk.Frame(root)
metrics_frame.pack(pady=10)

area_var = tk.StringVar()
modules_var = tk.StringVar()
status_var = tk.StringVar()

tk.Label(metrics_frame, textvariable=area_var, font=("Arial", 12)).grid(row=0, column=0, padx=10)
tk.Label(metrics_frame, textvariable=modules_var, font=("Arial", 12)).grid(row=0, column=1, padx=10)
tk.Label(metrics_frame, textvariable=status_var, font=("Arial", 12)).grid(row=0, column=2, padx=10)

update_metrics()

evaluate_button = tk.Button(root, text="Evaluate", command=evaluate_habitat, bg="lightblue", font=("Arial", 12))
evaluate_button.pack(pady=10)

root.mainloop()
