# app.py
# Requirements: pip install flask opencv-python ultralytics numpy torch torchvision gymnasium
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
from rl_agent import get_rl_action

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
current_phase = 0            # 0: Lane A (North-South) Green, 1: Lane B (East-West) Green
total_crossed = 0            # Cumulative vehicles crossed the counting line
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

# Tracking history for flow counting (Phase 1)
vehicle_tracks = {}         # track_id -> centroid list
counted_vehicles = set()    # track_ids that have crossed the line

# ── Traffic Simulator Class for Fallback/Simulation Mode ──
class TrafficSimulator:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.vehicles = []
        self.classes = ["car", "motorcycle", "bus", "truck", "ambulence", "fire_truck"]
        self.colors = {
            "car": (16, 185, 129),       # Green BGR: (129, 185, 16)
            "motorcycle": (250, 139, 167), # Purple BGR: (250, 139, 167)
            "bus": (153, 72, 236),       # Pink BGR: (153, 72, 236)
            "truck": (11, 158, 245),     # Amber BGR: (11, 158, 245)
            "ambulence": (0, 0, 255),    # Red
            "fire_truck": (0, 0, 255)    # Red
        }
        self.frame_count = 0
        
    def update(self):
        global total_crossed
        self.frame_count += 1
        
        # Spawn new vehicles randomly based on green phase
        # If Phase 0 is green, spawn more in Lane A (flowing), otherwise spawn in Lane B
        spawn_prob = 0.35
        if len(self.vehicles) < 10 and random.random() < spawn_prob:
            v_type = random.choices(
                self.classes, 
                weights=[0.60, 0.20, 0.08, 0.08, 0.02, 0.02], 
                k=1
            )[0]
            
            direction = random.choice([1, -1]) # 1: top->down (A), -1: bottom->up (B)
            if direction == 1:
                y = 0
                x = random.randint(int(self.width * 0.55), int(self.width * 0.85))
                # If Lane B is green, vehicles in Lane A slow down/stop at red light (y = 150)
                speed = random.randint(5, 8)
            else:
                y = self.height
                x = random.randint(int(self.width * 0.15), int(self.width * 0.45))
                speed = random.randint(-8, -5)
                
            self.vehicles.append({
                "id": random.randint(100, 999),
                "x": x,
                "y": y,
                "speed": speed,
                "type": v_type,
                "crossed": False,
                "size": (35, 55) if v_type in ["car", "ambulence"] else ((18, 35) if v_type == "motorcycle" else (48, 85))
            })
            
        # Move vehicles and check line crossing at y = 240
        active_vehicles = []
        counts = {c: 0 for c in self.classes}
        emergency_detected = False
        
        # Intersection stop lines (simulating red lights)
        stop_line_down = int(self.height * 0.35) # y=168
        stop_line_up = int(self.height * 0.65)   # y=312
        
        for v in self.vehicles:
            speed = v["speed"]
            y = v["y"]
            
            # Simulate stopping at red lights
            if current_phase == 1 and speed > 0 and y < stop_line_down and (y + speed) >= stop_line_down:
                # Lane A has red light, vehicle stops
                y = stop_line_down
            elif current_phase == 0 and speed < 0 and y > stop_line_up and (y + speed) <= stop_line_up:
                # Lane B has red light, vehicle stops
                y = stop_line_up
            else:
                # Normal motion
                y += speed
                
            v["y"] = y
            
            # Keep if inside window padding
            if (speed > 0 and y < self.height + 80) or (speed < 0 and y > -80):
                active_vehicles.append(v)
                # Count if visible on screen
                if 0 <= y <= self.height:
                    counts[v["type"]] += 1
                    if v["type"] in ["ambulence", "fire_truck"]:
                        emergency_detected = True
                        
                    # Check line crossing flow (Phase 1)
                    # Green line is drawn at y = 240 (middle of the screen)
                    if not v["crossed"]:
                        if (speed > 0 and y >= 240) or (speed < 0 and y <= 240):
                            v["crossed"] = True
                            total_crossed += 1
                            
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
                
        # Draw Counting Line (Phase 1) - Bright green line at y = 240
        cv2.line(frame, (int(self.width * 0.1), 240), (int(self.width * 0.9), 240), (10, 185, 16), 2)
        cv2.putText(frame, "COUNTING LINE (FLOW)", (int(self.width * 0.12), 232), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 185, 16), 1)
        
        # Draw Stop Lines
        stop_color_down = (0, 0, 255) if current_phase == 1 else (0, 255, 0)
        stop_color_up = (0, 0, 255) if current_phase == 0 else (0, 255, 0)
        # Lane A stop line
        cv2.line(frame, (int(self.width * 0.5), 168), (int(self.width * 0.9), 168), stop_color_down, 3)
        # Lane B stop line
        cv2.line(frame, (int(self.width * 0.1), 312), (int(self.width * 0.5), 312), stop_color_up, 3)
        
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
            cv2.putText(frame, f"{v['type']} (ID:{v['id']})", (x - w//2, y - h//2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            
        return frame

def process_frame(frame):
    global emergency_alert, vehicle_counts, total_crossed, vehicle_tracks, counted_vehicles

    if model is None:
        return frame, 0

    # 1. Run YOLO object tracking (Phase 1)
    results = model.track(frame, persist=True, verbose=False)
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
    
    # Draw green counting line at y = 300 on camera
    line_y = int(frame.shape[0] * 0.6) # 60% down the screen
    cv2.line(annotated_frame, (0, line_y), (frame.shape[1], line_y), (10, 185, 16), 2)
    cv2.putText(annotated_frame, "COUNTING LINE (FLOW)", (20, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 185, 16), 2)

    if detections.id is not None:
        track_ids = detections.id.int().tolist()
        for idx, box in enumerate(detections):
            cls = int(box.cls[0])
            if cls in VEHICLE_CLASSES:
                label = VEHICLE_CLASSES[cls]
                track_id = track_ids[idx]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Centroid calculations
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # 2. Tracking History & Line Crossing Math (Phase 1)
                if track_id not in vehicle_tracks:
                    vehicle_tracks[track_id] = []
                
                prev_y = None
                if vehicle_tracks[track_id]:
                    prev_y = vehicle_tracks[track_id][-1][1]
                    
                vehicle_tracks[track_id].append((cx, cy))
                if len(vehicle_tracks[track_id]) > 30: # limit history
                    vehicle_tracks[track_id].pop(0)
                    
                # Flow check: crossing line_y
                if prev_y is not None and track_id not in counted_vehicles:
                    # Crossed from above to below or below to above
                    if (prev_y < line_y and cy >= line_y) or (prev_y > line_y and cy <= line_y):
                        total_crossed += 1
                        counted_vehicles.add(track_id)
                
                crop = frame[y1:y2, x1:x2]

                # Check emergency classification
                if label in ["car", "bus", "truck"]:
                    try:
                        filename = f"temp_{label}_{track_id}.png"
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
                box_color = (129, 185, 16) if label == "car" else ((250, 139, 167) if label == "motorcycle" else (11, 158, 245))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(annotated_frame, f"{label} (ID:{track_id})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

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

    return annotated_frame, total

# ── Background Thread Worker for Asynchronous Camera Capture ──
def camera_worker():
    global streaming_active, simulation_mode, vehicle_counts, emergency_alert, signal_time, recent_totals, avg_total_vehicles, latest_frame_encoded, current_phase, total_crossed, vehicle_tracks, counted_vehicles
    
    print(f"[Background Thread] Initializing camera streams (SimulationMode={simulation_mode})...")
    cap = None
    
    # Reset tracking state
    vehicle_tracks = {}
    counted_vehicles = set()
    total_crossed = 0
    current_phase = 0
    signal_time = 5
    
    if not simulation_mode:
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
    
    last_history_update = time.time()
    last_rl_update = time.time()
    
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
            time.sleep(0.02)
            
        # 3. Reinforcement Learning Phase Decision (Phase 3)
        # Every 5 seconds, query the DQN agent for an optimization decision
        current_time = time.time()
        
        # Calculate simulated queue lengths based on current count distributions
        # Lane A: representing North-South. Lane B: representing East-West
        # In real stream, we can map vehicles detected on left half (Lane B) vs right half (Lane A)
        queue_a = vehicle_counts["car"] + vehicle_counts["truck"]
        queue_b = vehicle_counts["two_wheeler"] + vehicle_counts["bus"]
        
        if current_time - last_rl_update >= 5.0:
            if emergency_alert:
                # Emergency override corridor (forces green on the emergency vehicle lane)
                # If ambulance/firetruck is detected, override DQN to clear the congestion
                print("[RL Agent] Emergency vehicle override active. Forcing green corridor.")
                current_phase = 0 # Corresponds to Lane A corridor
                signal_time = 1
            else:
                # Query PyTorch DQN Agent
                action = get_rl_action(queue_a, queue_b, current_phase)
                if action == 1:
                    print(f"[RL Agent] State: A={queue_a}, B={queue_b}, Phase={current_phase} -> Decision: SWITCH PHASE")
                    current_phase = 1 - current_phase # Toggle green phase
                else:
                    print(f"[RL Agent] State: A={queue_a}, B={queue_b}, Phase={current_phase} -> Decision: KEEP PHASE")
                
                # Reset countdown timer
                signal_time = 5
            last_rl_update = current_time
        else:
            # Decrement countdown timer
            elapsed = int(current_time - last_rl_update)
            signal_time = max(1, 5 - elapsed)
            
        # Update history queue for LSTM forecasting at steady intervals
        if current_time - last_history_update >= 2.5:
            recent_totals.append(total_in_frame)
            if len(recent_totals) > MAX_HISTORY:
                recent_totals.pop(0)
            
            if len(recent_totals) == 10:
                avg_total_vehicles = predict_next_hour(recent_totals)
            else:
                avg_total_vehicles = sum(recent_totals) / len(recent_totals) if recent_totals else 0.0
                
            last_history_update = current_time
            
        if frame is not None:
            # Overlay active signal indicator on frame (top right corner)
            phase_label = "LANE A (N-S) GREEN" if current_phase == 0 else "LANE B (E-W) GREEN"
            phase_color = (129, 185, 16) if current_phase == 0 else (11, 158, 245)
            cv2.rectangle(frame, (360, 10), (620, 45), (15, 23, 42), -1)
            cv2.rectangle(frame, (360, 10), (620, 45), phase_color, 1)
            cv2.putText(frame, phase_label, (375, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_color, 2)
            
            # Draw flow counter overlay on frame
            cv2.rectangle(frame, (10, 120), (220, 155), (15, 23, 42), -1)
            cv2.rectangle(frame, (10, 120), (220, 155), (255, 255, 255), 1)
            cv2.putText(frame, f"Crossed Flow: {total_crossed}", (20, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            ret, jpeg = cv2.imencode(".jpg", frame)
            if ret:
                with frame_lock:
                    latest_frame_encoded = jpeg.tobytes()
                    
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
    global streaming_active, camera_thread, vehicle_counts, emergency_alert, signal_time, avg_total_vehicles, total_crossed
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
    total_crossed = 0
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
        "avg_total_vehicles": round(avg_total_vehicles, 1),
        "current_phase": current_phase,
        "total_crossed": total_crossed
    })


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True, port=5000)