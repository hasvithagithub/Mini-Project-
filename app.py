# app.py
# Requirements: pip install flask opencv-python ultralytics numpy torch torchvision
import cv2
import numpy as np
import os
import time
import random
import threading
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
from model_predict2 import pred_emergency_vehicle
from predicter import predict_next_hour

app = Flask(__name__)

# Load YOLO model (loaded globally at startup)
print("Loading YOLOv8 model...")
try:
    model = YOLO("yolov8n.pt")
    print("YOLOv8 loaded successfully.")
except Exception as e:
    print(f"Warning: YOLOv8 model loading failed: {e}. Running in simulation/mock mode only.")
    model = None

# COCO vehicle class IDs
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Threading & Streaming state
streaming_active = False
camera_thread = None
latest_frame_encoded = None
frame_lock = threading.Lock()

# Traffic state variables (shared across threads, read by /stats)
emergency_alert = False
signal_time = 5
vehicle_counts = {
    "total": 0,
    "two_wheeler": 0,
    "car": 0,
    "bus": 0,
    "truck": 0,
    "ambulence": 0,
    "fire_truck": 0
}

recent_totals = []          # keeps last 10 hourly equivalent counts
MAX_HISTORY = 10
avg_total_vehicles = 0.0    # displayed predicted value
simulation_mode = False

# ── Traffic Simulator Class for Fallback/Simulation Mode ──
class TrafficSimulator:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.vehicles = []
        self.classes = ["car", "motorcycle", "bus", "truck", "ambulence", "fire_truck"]
        self.colors = {
            "car": (16, 185, 129),       # #10B981 (Green) - reversed to BGR: (129, 185, 16)
            "motorcycle": (250, 139, 167), # #A78BFA (Purple) - BGR: (250, 139, 167)
            "bus": (153, 72, 236),       # #EC4899 (Pink) - BGR: (153, 72, 236)
            "truck": (11, 158, 245),     # #F59E0B (Amber) - BGR: (11, 158, 245)
            "ambulence": (0, 0, 255),    # Red
            "fire_truck": (0, 0, 255)    # Red
        }
        self.frame_count = 0
        
    def update(self):
        self.frame_count += 1
        # Spawn new vehicles randomly (higher probability so traffic keeps moving)
        if len(self.vehicles) < 10 and random.random() < 0.35:
            v_type = random.choices(
                self.classes, 
                weights=[0.60, 0.20, 0.08, 0.08, 0.02, 0.02], 
                k=1
            )[0]
            
            # Start position: direction 1 (top of screen moving down) or -1 (bottom moving up)
            direction = random.choice([1, -1])
            if direction == 1:
                y = 0
                x = random.randint(int(self.width * 0.55), int(self.width * 0.85))
                speed = random.randint(5, 10)
            else:
                y = self.height
                x = random.randint(int(self.width * 0.15), int(self.width * 0.45))
                speed = random.randint(-10, -5)
                
            self.vehicles.append({
                "x": x,
                "y": y,
                "speed": speed,
                "type": v_type,
                "size": (35, 55) if v_type in ["car", "ambulence"] else ((18, 35) if v_type == "motorcycle" else (48, 85))
            })
            
        # Move vehicles
        active_vehicles = []
        counts = {c: 0 for c in self.classes}
        emergency_detected = False
        
        for v in self.vehicles:
            v["y"] += v["speed"]
            # Keep if inside window padding
            if (v["speed"] > 0 and v["y"] < self.height + 80) or (v["speed"] < 0 and v["y"] > -80):
                active_vehicles.append(v)
                # Count if visible on screen
                if 0 <= v["y"] <= self.height:
                    counts[v["type"]] += 1
                    if v["type"] in ["ambulence", "fire_truck"]:
                        emergency_detected = True
                        
        self.vehicles = active_vehicles
        return counts, emergency_detected
        
    def draw_frame(self):
        # Base background - Dark Charcoal (#111827)
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :, 0] = 39  # B
        frame[:, :, 1] = 24  # G
        frame[:, :, 2] = 17  # R
        
        # Draw road bounds
        cv2.rectangle(frame, (int(self.width * 0.1), 0), (int(self.width * 0.9), self.height), (55, 65, 81), -1) # Asphalt gray
        
        # Center lane dividing line
        cv2.line(frame, (int(self.width * 0.5), 0), (int(self.width * 0.5), self.height), (0, 215, 255), 3) # Yellow line
        
        # Left and right boundary lines
        cv2.line(frame, (int(self.width * 0.1), 0), (int(self.width * 0.1), self.height), (209, 213, 219), 2)
        cv2.line(frame, (int(self.width * 0.9), 0), (int(self.width * 0.9), self.height), (209, 213, 219), 2)
        
        # Dashed dividers
        for y in range(0, self.height, 40):
            if (y // 40) % 2 == 0:
                cv2.line(frame, (int(self.width * 0.3), y), (int(self.width * 0.3), y + 20), (156, 163, 175), 1)
                cv2.line(frame, (int(self.width * 0.7), y), (int(self.width * 0.7), y + 20), (156, 163, 175), 1)
                
        # Draw vehicles
        for v in self.vehicles:
            x, y = v["x"], v["y"]
            w, h = v["size"]
            color = self.colors[v["type"]]
            
            # Vehicle shadow/wheels
            cv2.rectangle(frame, (x - w//2 - 2, y - h//2), (x - w//2 + 4, y - h//4), (0,0,0), -1)
            cv2.rectangle(frame, (x + w//2 - 4, y - h//2), (x + w//2 + 2, y - h//4), (0,0,0), -1)
            cv2.rectangle(frame, (x - w//2 - 2, y + h//4), (x - w//2 + 4, y + h//2), (0,0,0), -1)
            cv2.rectangle(frame, (x + w//2 - 4, y + h//4), (x + w//2 + 2, y + h//2), (0,0,0), -1)
            
            # Vehicle chassis
            cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), color, -1)
            
            # Windshield/cabin
            cv2.rectangle(frame, (x - w//2 + 3, y - h//3), (x + w//2 - 3, y - h//4), (75, 85, 99), -1)
            cv2.rectangle(frame, (x - w//2 + 3, y + h//4), (x + w//2 - 3, y + h//3), (75, 85, 99), -1)
            
            # Flashing lights for ambulance/fire truck
            if v["type"] in ["ambulence", "fire_truck"]:
                light_color = (0, 0, 255) if (self.frame_count // 3) % 2 == 0 else (255, 0, 0)
                cv2.circle(frame, (x, y - h//6), 6, light_color, -1)
                cv2.circle(frame, (x, y + h//6), 6, (255, 255, 255), -1)
                
            # Bounding box & text overlays (simulates live YOLO detections)
            cv2.rectangle(frame, (x - w//2 - 4, y - h//2 - 4), (x + w//2 + 4, y + h//2 + 4), color, 2)
            cv2.putText(frame, f"{v['type']}", (x - w//2, y - h//2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            
        return frame

def process_frame(frame):
    global emergency_alert, signal_time, vehicle_counts, recent_totals, avg_total_vehicles

    if model is None:
        return frame, 0

    results = model(frame, verbose=False)
    detections = results[0].boxes

    counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0,
        "ambulence": 0,
        "fire_truck": 0
    }

    emergency_alert = False
    annotated_frame = frame.copy()

    for idx, box in enumerate(detections):
        cls = int(box.cls[0])
        if cls in VEHICLE_CLASSES:
            label = VEHICLE_CLASSES[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            # Check emergency classification for car, bus, truck
            if label in ["car", "bus", "truck"]:
                try:
                    filename = f"temp_{label}_{idx}.png"
                    cv2.imwrite(filename, crop)
                    pred_label, conf = pred_emergency_vehicle(filename)
                    if os.path.exists(filename):
                        os.remove(filename)

                    if pred_label in ["ambulence", "fire_truck"] and conf > 0.7:
                        counts[pred_label] += 1
                        emergency_alert = True
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(annotated_frame, f"{pred_label} ({conf*100:.1f}%)",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        continue
                except Exception as e:
                    print("Emergency model error:", e)

            # Normal vehicle counting
            counts[label] += 1
            box_color = (16, 185, 129) if label == "car" else ((250, 139, 167) if label == "motorcycle" else (11, 158, 245))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(annotated_frame, label,
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    total = sum(counts.values())

    # Update global counts
    vehicle_counts = {
        "total": total,
        "two_wheeler": counts["motorcycle"],
        "car": counts["car"],
        "bus": counts["bus"],
        "truck": counts["truck"],
        "ambulence": counts["ambulence"],
        "fire_truck": counts["fire_truck"]
    }

    # Signal timing logic
    if emergency_alert:
        signal_time = 1
    elif total > 15:
        signal_time = 2
    else:
        signal_time = 5

    return annotated_frame, total

# ── Background Thread Worker for Asynchronous Camera Capture ──
def camera_worker():
    global streaming_active, simulation_mode, vehicle_counts, emergency_alert, signal_time, recent_totals, avg_total_vehicles, latest_frame_encoded
    
    print(f"[Background Thread] Initializing camera streams (SimulationMode={simulation_mode})...")
    cap = None
    
    if not simulation_mode:
        # Try IP camera
        try:
            print("[Background Thread] Connecting to IP camera...")
            cap = cv2.VideoCapture("http://192.0.0.4:8080/video")
            if cap is None or not cap.isOpened():
                print("[Background Thread] IP camera connection failed. Attempting local webcam (index 0)...")
                if cap: cap.release()
                cap = cv2.VideoCapture(0)
        except Exception as e:
            print(f"[Background Thread] Error checking camera hardware: {e}")
            
        if cap is None or not cap.isOpened():
            print("[Background Thread] No camera hardware available. Switching to Fallback Simulator.")
            simulation_mode = True
            if cap:
                cap.release()
                cap = None
                
    if simulation_mode:
        print("[Background Thread] Launching Traffic Simulator.")
        simulator = TrafficSimulator()
        
    recent_totals = []
    avg_total_vehicles = 0.0
    
    # Track time to add a count to the prediction history list
    last_history_update = time.time()
    
    while streaming_active:
        frame = None
        total_in_frame = 0
        
        if simulation_mode:
            # 1. Run Simulator
            counts, alert = simulator.update()
            frame = simulator.draw_frame()
            
            total_in_frame = sum(counts.values())
            vehicle_counts = {
                "total": total_in_frame,
                "two_wheeler": counts["motorcycle"],
                "car": counts["car"],
                "bus": counts["bus"],
                "truck": counts["truck"],
                "ambulence": counts["ambulence"],
                "fire_truck": counts["fire_truck"]
            }
            emergency_alert = alert
            
            if emergency_alert:
                signal_time = 1
            elif total_in_frame > 6:
                signal_time = 2
            else:
                signal_time = 5
                
            # Telemetry text overlays on simulated stream
            cv2.putText(frame, f"Total Vehicles: {total_in_frame}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"LSTM Predicted (1h): {avg_total_vehicles:.1f}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 235, 59), 2)
            if emergency_alert:
                cv2.putText(frame, "EMERGENCY OVERRIDE", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
            time.sleep(0.08) # 12.5 FPS
        else:
            # 2. Run Real Camera Stream
            ret, raw_frame = cap.read()
            if not ret:
                print("[Background Thread] Stream read failed. Falling back to Simulator.")
                simulation_mode = True
                simulator = TrafficSimulator()
                if cap:
                    cap.release()
                    cap = None
                continue
                
            frame, total_in_frame = process_frame(raw_frame)
            time.sleep(0.02) # throttle slightly
            
        # Update history queue for LSTM forecasting at steady intervals (every 2.5 seconds, simulating hourly readings)
        current_time = time.time()
        if current_time - last_history_update >= 2.5:
            recent_totals.append(total_in_frame)
            if len(recent_totals) > MAX_HISTORY:
                recent_totals.pop(0)
            
            # Predict using LSTM
            if len(recent_totals) == 10:
                avg_total_vehicles = predict_next_hour(recent_totals)
            else:
                avg_total_vehicles = sum(recent_totals) / len(recent_totals) if recent_totals else 0.0
                
            last_history_update = current_time
            
        if frame is not None:
            ret, jpeg = cv2.imencode(".jpg", frame)
            if ret:
                with frame_lock:
                    latest_frame_encoded = jpeg.tobytes()
                    
    # Clean up
    if cap:
        cap.release()
        cap = None
    print("[Background Thread] Camera worker thread terminated.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/camera")
def camera():
    return render_template("camera.html")


@app.route("/start_camera")
def start_camera():
    global streaming_active, camera_thread, simulation_mode
    from flask import request
    mode = request.args.get('mode', 'simulation')
    
    # If mode switched while running, stop first
    if streaming_active:
        if (mode == 'simulation' and not simulation_mode) or (mode == 'camera' and simulation_mode):
            print("Mode changed while streaming. Restarting worker thread...")
            stop_camera()
            time.sleep(0.5)
            
    if not streaming_active:
        simulation_mode = (mode == 'simulation')
        streaming_active = True
        camera_thread = threading.Thread(target=camera_worker)
        camera_thread.daemon = True
        camera_thread.start()
        print(f"Camera worker thread spawned successfully in mode: {mode}.")
        
    return "Camera started"


@app.route("/stop_camera")
def stop_camera():
    global streaming_active, camera_thread, vehicle_counts, emergency_alert, signal_time, avg_total_vehicles
    streaming_active = False
    if camera_thread:
        camera_thread.join(timeout=1.5)
        camera_thread = None
    
    # Reset stats
    vehicle_counts = {
        "total": 0,
        "two_wheeler": 0,
        "car": 0,
        "bus": 0,
        "truck": 0,
        "ambulence": 0,
        "fire_truck": 0
    }
    emergency_alert = False
    signal_time = 5
    avg_total_vehicles = 0.0
    return "Camera stopped"


@app.route("/video_feed")
def video_feed():
    def gen():
        global streaming_active, latest_frame_encoded
        last_yielded_frame = None
        while streaming_active:
            frame_bytes = None
            with frame_lock:
                frame_bytes = latest_frame_encoded
                
            if frame_bytes is not None and frame_bytes != last_yielded_frame:
                last_yielded_frame = frame_bytes
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       frame_bytes + b"\r\n")
            else:
                time.sleep(0.01)
                
    if not streaming_active:
        return "Camera not started", 400
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    return jsonify({
        "counts": vehicle_counts,
        "signal_time": signal_time,
        "alert": emergency_alert,
        "avg_total_vehicles": round(avg_total_vehicles, 1)
    })


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True, port=5000)