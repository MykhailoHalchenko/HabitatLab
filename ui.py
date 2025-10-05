import tkinter as tk
from tkinter import messagebox
import main
from datacom.network_topology import NetworkTopology
from datacom.storage_manager import save_plan_to_json, load_plan_from_json, export_metrics_to_csv, plan_from_network_topology
canvas_nodes = []

def convert_canvas_to_topology():
    topology = NetworkTopology()
    for node in canvas_nodes:
        topology.add_module(node["id"], node["type"], node["position"])
    topology.create_graph()
    return topology

def finalize_rectangle(event):
    x1, y1 = canvas.start_x, canvas.start_y
    x2, y2 = event.x, event.y
    rect = canvas.create_rectangle(x1, y1, x2, y2, outline="black", width=2)

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    node_id = f"module_{len(canvas_nodes)+1}"
    node_type = "generic"

    canvas_nodes.append({
        "id": node_id,
        "type": node_type,
        "position": (center_x, center_y)
    })

def evaluate_habitat():
    main.main(convert_canvas_to_topology())
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
    
def save_current_network_to_plan(network, filename=None):
    plan = plan_from_network_topology(network)
    path = save_plan_to_json(plan, filename)
    print(f"Plan saved to: {path}")
    return path

def export_simulation_metrics(metrics, filename=None):
    path = export_metrics_to_csv(metrics, filename)
    print(f"Metrics exported to: {path}")
    return path
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

