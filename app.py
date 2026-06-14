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

# New global state variables for interactivity
control_mode = "dqn"         # "dqn" or "manual"
manual_override_phase = 0    # 0: Lane A green, 1: Lane B green
dqn_logs = []
MAX_DQN_LOGS = 15
spawn_rate = 0.35            # probability of spawning in simulation (0.0 - 1.0)
simulated_emergencies = []   # list of emergency vehicles to spawn on demand
dqn_cycle_time = 5           # DQN Optimizer decision cycle time (seconds)
simulation_speed_limit = 8   # Max vehicle speed (3 to 15 pixels/frame)

def log_dqn_event(message):
    global dqn_logs
    timestamp = time.strftime("%H:%M:%S")
    dqn_logs.append(f"[{timestamp}] {message}")
    if len(dqn_logs) > MAX_DQN_LOGS:
        dqn_logs.pop(0)

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
        self.camera_view_mode = "top-down"
        
    def update(self):
        global total_crossed, current_phase, simulated_emergencies, spawn_rate, simulation_speed_limit
        self.frame_count += 1
        
        # Toggle camera view every 15 seconds
        current_time = time.time()
        if int(current_time) % 30 < 15:
            self.camera_view_mode = "top-down"
        else:
            self.camera_view_mode = "angled"

        # Spawn new vehicles randomly based on spawn rate
        v_type = None
        if simulated_emergencies:
            v_type = simulated_emergencies.pop(0)
        elif len(self.vehicles) < 16 and random.random() < spawn_rate:
            v_type = random.choices(
                self.classes, 
                weights=[0.60, 0.20, 0.08, 0.08, 0.02, 0.02], 
                k=1
            )[0]
        
        if v_type is not None:
            axis = random.choice(["v", "h"])
            direction = random.choice([1, -1])
            
            if axis == "v":
                if direction == 1: # Southbound (top -> down)
                    x = random.randint(260, 300)
                    y = 0
                    speed = random.randint(max(3, simulation_speed_limit - 3), simulation_speed_limit)
                else: # Northbound (bottom -> up)
                    x = random.randint(340, 380)
                    y = self.height
                    speed = -random.randint(max(3, simulation_speed_limit - 3), simulation_speed_limit)
            else:
                if direction == 1: # Eastbound (left -> right)
                    x = 0
                    y = random.randint(260, 300)
                    speed = random.randint(max(3, simulation_speed_limit - 3), simulation_speed_limit)
                else: # Westbound (right -> left)
                    x = self.width
                    y = random.randint(180, 220)
                    speed = -random.randint(max(3, simulation_speed_limit - 3), simulation_speed_limit)
            
            # Base sizes (w, h) for vertical
            if v_type in ["car", "ambulence"]:
                base_size = (30, 48)
            elif v_type == "motorcycle":
                base_size = (16, 28)
            else: # bus or truck
                base_size = (42, 75)
                
            # If horizontal, swap width and height
            size = base_size if axis == "v" else (base_size[1], base_size[0])
            
            self.vehicles.append({
                "id": random.randint(100, 999),
                "x": x,
                "y": y,
                "speed": speed,
                "type": v_type,
                "axis": axis,
                "direction": direction,
                "crossed": False,
                "size": size
            })
            
        # Group and sort vehicles by lane to apply queuing logic
        lanes = {
            "sb": [], # Southbound
            "nb": [], # Northbound
            "eb": [], # Eastbound
            "wb": []  # Westbound
        }
        for v in self.vehicles:
            if v["axis"] == "v":
                if v["speed"] > 0:
                    lanes["sb"].append(v)
                else:
                    lanes["nb"].append(v)
            else:
                if v["speed"] > 0:
                    lanes["eb"].append(v)
                else:
                    lanes["wb"].append(v)
                    
        lanes["sb"].sort(key=lambda v: v["y"], reverse=True)
        lanes["nb"].sort(key=lambda v: v["y"], reverse=False)
        lanes["eb"].sort(key=lambda v: v["x"], reverse=True)
        lanes["wb"].sort(key=lambda v: v["x"], reverse=False)
        
        active_vehicles = []
        counts = {c: 0 for c in self.classes}
        emergency_detected = False
        
        # 1. Southbound Queue Update
        for i, v in enumerate(lanes["sb"]):
            speed = min(max(3, simulation_speed_limit - 3), simulation_speed_limit)
            v["speed"] = speed
            stop_y = 160
            can_move = True
            
            # Stop if red light (Phase 1) and above the stop line
            if current_phase == 1 and v["y"] <= stop_y and v["type"] not in ["ambulence", "fire_truck"]:
                if v["y"] + speed >= stop_y:
                    v["y"] = stop_y
                    can_move = False
            
            if can_move:
                if i > 0 and v["type"] not in ["ambulence", "fire_truck"]:
                    v_ahead = lanes["sb"][i-1]
                    safety_dist = v_ahead["size"][1] + 15
                    if v["y"] + speed + safety_dist >= v_ahead["y"]:
                        v["y"] = max(v["y"], v_ahead["y"] - safety_dist)
                    else:
                        v["y"] += speed
                else:
                    v["y"] += speed
            
            if v["y"] < self.height + 80:
                active_vehicles.append(v)
                
        # 2. Northbound Queue Update
        for i, v in enumerate(lanes["nb"]):
            speed = -min(max(3, simulation_speed_limit - 3), simulation_speed_limit)
            v["speed"] = speed
            stop_y = 320
            can_move = True
            
            # Stop if red light (Phase 1) and below the stop line
            if current_phase == 1 and v["y"] >= stop_y and v["type"] not in ["ambulence", "fire_truck"]:
                if v["y"] + speed <= stop_y:
                    v["y"] = stop_y
                    can_move = False
            
            if can_move:
                if i > 0 and v["type"] not in ["ambulence", "fire_truck"]:
                    v_ahead = lanes["nb"][i-1]
                    safety_dist = v_ahead["size"][1] + 15
                    if v["y"] + speed - safety_dist <= v_ahead["y"]:
                        v["y"] = min(v["y"], v_ahead["y"] + safety_dist)
                    else:
                        v["y"] += speed
                else:
                    v["y"] += speed
            
            if v["y"] > -80:
                active_vehicles.append(v)
                
        # 3. Eastbound Queue Update
        for i, v in enumerate(lanes["eb"]):
            speed = min(max(3, simulation_speed_limit - 3), simulation_speed_limit)
            v["speed"] = speed
            stop_x = 240
            can_move = True
            
            # Stop if red light (Phase 0) and left of stop line
            if current_phase == 0 and v["x"] <= stop_x and v["type"] not in ["ambulence", "fire_truck"]:
                if v["x"] + speed >= stop_x:
                    v["x"] = stop_x
                    can_move = False
            
            if can_move:
                if i > 0 and v["type"] not in ["ambulence", "fire_truck"]:
                    v_ahead = lanes["eb"][i-1]
                    safety_dist = v_ahead["size"][0] + 15
                    if v["x"] + speed + safety_dist >= v_ahead["x"]:
                        v["x"] = max(v["x"], v_ahead["x"] - safety_dist)
                    else:
                        v["x"] += speed
                else:
                    v["x"] += speed
            
            if v["x"] < self.width + 80:
                active_vehicles.append(v)
                
        # 4. Westbound Queue Update
        for i, v in enumerate(lanes["wb"]):
            speed = -min(max(3, simulation_speed_limit - 3), simulation_speed_limit)
            v["speed"] = speed
            stop_x = 400
            can_move = True
            
            # Stop if red light (Phase 0) and right of stop line
            if current_phase == 0 and v["x"] >= stop_x and v["type"] not in ["ambulence", "fire_truck"]:
                if v["x"] + speed <= stop_x:
                    v["x"] = stop_x
                    can_move = False
            
            if can_move:
                if i > 0 and v["type"] not in ["ambulence", "fire_truck"]:
                    v_ahead = lanes["wb"][i-1]
                    safety_dist = v_ahead["size"][0] + 15
                    if v["x"] + speed - safety_dist <= v_ahead["x"]:
                        v["x"] = min(v["x"], v_ahead["x"] + safety_dist)
                    else:
                        v["x"] += speed
                else:
                    v["x"] += speed
            
            if v["x"] > -80:
                active_vehicles.append(v)
                
        # Update crossed counts & check visible screen coordinates
        for v in active_vehicles:
            x, y = v["x"], v["y"]
            
            if 0 <= x <= self.width and 0 <= y <= self.height:
                counts[v["type"]] += 1
                if v["type"] in ["ambulence", "fire_truck"]:
                    emergency_detected = True
                    
            if not v["crossed"]:
                if v["axis"] == "v":
                    if (v["speed"] > 0 and y >= 160) or (v["speed"] < 0 and y <= 320):
                        v["crossed"] = True
                        total_crossed += 1
                else:
                    if (v["speed"] > 0 and x >= 240) or (v["speed"] < 0 and x <= 400):
                        v["crossed"] = True
                        total_crossed += 1
                        
        self.vehicles = active_vehicles
        return counts, emergency_detected

    def draw_frame(self):
        # Base canvas - Dark Charcoal
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :, 0] = 39  # B
        frame[:, :, 1] = 24  # G
        frame[:, :, 2] = 17  # R
        
        # 1. Draw Roads (Asphalt Gray)
        # Vertical Road (N-S)
        cv2.rectangle(frame, (240, 0), (400, self.height), (55, 65, 81), -1)
        # Horizontal Road (E-W)
        cv2.rectangle(frame, (0, 160), (self.width, 320), (55, 65, 81), -1)
        
        # Center Intersection Square overlap cleanup
        cv2.rectangle(frame, (240, 160), (400, 320), (55, 65, 81), -1)
        
        # 2. Draw Yellow Lane Dividers
        # N-S Yellow Divider lines
        cv2.line(frame, (320, 0), (320, 160), (0, 215, 255), 2)
        cv2.line(frame, (320, 320), (320, self.height), (0, 215, 255), 2)
        # E-W Yellow Divider lines
        cv2.line(frame, (0, 240), (240, 240), (0, 215, 255), 2)
        cv2.line(frame, (400, 240), (self.width, 240), (0, 215, 255), 2)
        
        # 3. Draw White Boundary Lines
        # N-S borders
        cv2.line(frame, (240, 0), (240, 160), (209, 213, 219), 2)
        cv2.line(frame, (240, 320), (240, self.height), (209, 213, 219), 2)
        cv2.line(frame, (400, 0), (400, 160), (209, 213, 219), 2)
        cv2.line(frame, (400, 320), (400, self.height), (209, 213, 219), 2)
        # E-W borders
        cv2.line(frame, (0, 160), (240, 160), (209, 213, 219), 2)
        cv2.line(frame, (400, 160), (self.width, 160), (209, 213, 219), 2)
        cv2.line(frame, (0, 320), (240, 320), (209, 213, 219), 2)
        cv2.line(frame, (400, 320), (self.width, 320), (209, 213, 219), 2)
        
        # Dashed dividers
        for y in range(0, 160, 30):
            cv2.line(frame, (280, y), (280, y + 15), (156, 163, 175), 1)
            cv2.line(frame, (360, y), (360, y + 15), (156, 163, 175), 1)
        for y in range(320, self.height, 30):
            cv2.line(frame, (280, y), (280, y + 15), (156, 163, 175), 1)
            cv2.line(frame, (360, y), (360, y + 15), (156, 163, 175), 1)
        for x in range(0, 240, 30):
            cv2.line(frame, (x, 200), (x + 15, 200), (156, 163, 175), 1)
            cv2.line(frame, (x, 280), (x + 15, 280), (156, 163, 175), 1)
        for x in range(400, self.width, 30):
            cv2.line(frame, (x, 200), (x + 15, 200), (156, 163, 175), 1)
            cv2.line(frame, (x, 280), (x + 15, 280), (156, 163, 175), 1)

        # 4. Draw Counting Lines (Green entry indicators)
        # N-S Entries
        cv2.line(frame, (240, 158), (320, 158), (10, 185, 16), 2)
        cv2.line(frame, (320, 322), (400, 322), (10, 185, 16), 2)
        # E-W Entries
        cv2.line(frame, (238, 240), (238, 320), (10, 185, 16), 2)
        cv2.line(frame, (402, 160), (402, 240), (10, 185, 16), 2)
        
        # 5. Draw Traffic Lights/Stop Lines
        color_ns = (0, 255, 0) if current_phase == 0 else (0, 0, 255)
        color_ew = (0, 255, 0) if current_phase == 1 else (0, 0, 255)
        
        # Southbound Stop
        cv2.line(frame, (240, 160), (320, 160), color_ns, 3)
        # Northbound Stop
        cv2.line(frame, (320, 320), (400, 320), color_ns, 3)
        # Eastbound Stop
        cv2.line(frame, (240, 240), (240, 320), color_ew, 3)
        # Westbound Stop
        cv2.line(frame, (400, 160), (400, 240), color_ew, 3)

        # 6. Draw Vehicles
        for v in self.vehicles:
            x, y = v["x"], v["y"]
            w, h = v["size"]
            color = self.colors[v["type"]]
            
            # Simple wheels/shadow
            if v["axis"] == "v":
                cv2.rectangle(frame, (x - w//2 - 2, y - h//2), (x - w//2 + 4, y - h//4), (0,0,0), -1)
                cv2.rectangle(frame, (x + w//2 - 4, y - h//2), (x + w//2 + 2, y - h//4), (0,0,0), -1)
                cv2.rectangle(frame, (x - w//2 - 2, y + h//4), (x - w//2 + 4, y + h//2), (0,0,0), -1)
                cv2.rectangle(frame, (x + w//2 - 4, y + h//4), (x + w//2 + 2, y + h//2), (0,0,0), -1)
            else:
                cv2.rectangle(frame, (x - w//2, y - h//2 - 2), (x - w//4, y - h//2 + 4), (0,0,0), -1)
                cv2.rectangle(frame, (x + h//4, y - h//2 - 2), (x + w//2, y - h//2 + 4), (0,0,0), -1)
                cv2.rectangle(frame, (x - w//2, y + h//2 - 4), (x - w//4, y + h//2 + 2), (0,0,0), -1)
                cv2.rectangle(frame, (x + h//4, y + h//2 - 4), (x + w//2, y + h//2 + 2), (0,0,0), -1)
                
            # Chassis
            cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), color, -1)
            
            # Windshield
            if v["axis"] == "v":
                cv2.rectangle(frame, (x - w//2 + 3, y - h//3), (x + w//2 - 3, y - h//4), (75, 85, 99), -1)
                cv2.rectangle(frame, (x - w//2 + 3, y + h//4), (x + w//2 - 3, y + h//3), (75, 85, 99), -1)
            else:
                cv2.rectangle(frame, (x - w//3, y - h//2 + 3), (x - w//4, y + h//2 - 3), (75, 85, 99), -1)
                cv2.rectangle(frame, (x + w//4, y - h//2 + 3), (x + w//3, y + h//2 - 3), (75, 85, 99), -1)
                
            # Flashing light for emergency
            if v["type"] in ["ambulence", "fire_truck"]:
                light_color = (0, 0, 255) if (self.frame_count // 3) % 2 == 0 else (255, 0, 0)
                if v["axis"] == "v":
                    cv2.circle(frame, (x, y - h//6), 6, light_color, -1)
                    cv2.circle(frame, (x, y + h//6), 6, (255, 255, 255), -1)
                else:
                    cv2.circle(frame, (x - w//6, y), 6, light_color, -1)
                    cv2.circle(frame, (x + w//6, y), 6, (255, 255, 255), -1)
                    
            # Sim YOLO bbox & tag
            cv2.rectangle(frame, (x - w//2 - 2, y - h//2 - 2), (x + w//2 + 2, y + h//2 + 2), color, 1)
            cv2.putText(frame, f"{v['type']}", (x - w//2, y - h//2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # 7. Apply perspective transformation if angled mode is active
        if self.camera_view_mode == "angled":
            src_pts = np.float32([[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]])
            # Wide-angle corner perspective mapping
            dst_pts = np.float32([
                [80, 100],
                [560, 40],
                [600, 380],
                [40, 420]
            ])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            frame = cv2.warpPerspective(frame, M, (self.width, self.height), borderValue=(39, 24, 17))
            
        # Draw Camera View HUD Overlay
        cv2.rectangle(frame, (10, 10), (290, 45), (15, 23, 42), -1)
        cv2.rectangle(frame, (10, 10), (290, 45), (255, 255, 255), 1)
        hud_text = "VIEW: AERIAL TOP-DOWN (15s)" if self.camera_view_mode == "top-down" else "VIEW: ANGLED CCTV (15s)"
        cv2.putText(frame, hud_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 2)
        
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

    lane_a_count = 0
    lane_b_count = 0

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
                
                # Spatial lane classification (right side is Lane A, left side is Lane B)
                if cx > frame.shape[1] // 2:
                    lane_a_count += 1
                else:
                    lane_b_count += 1
                
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
        "fire_truck": counts["fire_truck"],
        "lane_a": lane_a_count,
        "lane_b": lane_b_count
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
    had_emergency = False
    
    while streaming_active:
        frame = None
        total_in_frame = 0
        
        if simulation_mode:
            # 1. Run Simulator
            counts, alert = simulator.update()
            frame = simulator.draw_frame()
            
            total_in_frame = sum(counts.values())
            
            # Count actual lanes in the simulator
            sim_lane_a = sum(1 for v in simulator.vehicles if v["axis"] == "v")
            sim_lane_b = sum(1 for v in simulator.vehicles if v["axis"] == "h")
            
            vehicle_counts = {
                "total": total_in_frame,
                "two_wheeler": counts["motorcycle"],
                "car": counts["car"],
                "bus": counts["bus"],
                "truck": counts["truck"],
                "ambulence": counts["ambulence"],
                "fire_truck": counts["fire_truck"],
                "lane_a": sim_lane_a,
                "lane_b": sim_lane_b
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
        
        # Calculate queue lengths based on spatial lanes
        queue_a = vehicle_counts.get("lane_a", 0)
        queue_b = vehicle_counts.get("lane_b", 0)
        
        if control_mode == "manual":
            current_phase = manual_override_phase
            signal_time = 99
            if current_time - last_rl_update >= dqn_cycle_time:
                log_dqn_event(f"MANUAL MODE: Active phase forced to Lane {'A (N-S)' if current_phase == 0 else 'B (E-W)'}")
                last_rl_update = current_time
            had_emergency = False
        else:
            if control_mode == "timer":
                if current_time - last_rl_update >= dqn_cycle_time:
                    current_phase = 1 - current_phase # Toggle green phase
                    signal_time = dqn_cycle_time
                    log_dqn_event(f"TIMER MODE: Automatically switched green phase to Lane {'A (N-S)' if current_phase == 0 else 'B (E-W)'}")
                    last_rl_update = current_time
                else:
                    elapsed = int(current_time - last_rl_update)
                    signal_time = max(1, dqn_cycle_time - elapsed)
            else: # dqn mode
                if current_time - last_rl_update >= dqn_cycle_time:
                    # Query PyTorch DQN Agent
                    action = get_rl_action(queue_a, queue_b, current_phase)
                    decision_str = ""
                    if action == 1:
                        current_phase = 1 - current_phase # Toggle green phase
                        decision_str = f"SWITCH PHASE to Lane {'A (N-S)' if current_phase == 0 else 'B (E-W)'}"
                    else:
                        decision_str = f"KEEP PHASE on Lane {'A (N-S)' if current_phase == 0 else 'B (E-W)'}"
                    
                    print(f"[RL Agent] State: A={queue_a}, B={queue_b}, Phase={current_phase} -> Decision: {decision_str}")
                    log_dqn_event(f"DQN Decision: {decision_str} (Queue A: {queue_a}, Queue B: {queue_b})")
                    
                    # Reset countdown timer
                    signal_time = dqn_cycle_time
                    last_rl_update = current_time
                else:
                    # Decrement countdown timer
                    elapsed = int(current_time - last_rl_update)
                    signal_time = max(1, dqn_cycle_time - elapsed)
            
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
        "total_crossed": total_crossed,
        "control_mode": control_mode,
        "dqn_logs": dqn_logs,
        "spawn_rate": spawn_rate,
        "dqn_cycle_time": dqn_cycle_time,
        "simulation_speed_limit": simulation_speed_limit
    })


@app.route("/set_control_mode")
def set_control_mode():
    global control_mode, manual_override_phase
    from flask import request
    mode = request.args.get("mode", "dqn")
    phase = request.args.get("phase", "0")
    if mode in ["dqn", "manual", "timer"]:
        control_mode = mode
    try:
        manual_override_phase = int(phase)
    except ValueError:
        pass
    return jsonify({
        "status": "success",
        "control_mode": control_mode,
        "manual_override_phase": manual_override_phase
    })


@app.route("/trigger_emergency")
def trigger_emergency():
    global simulated_emergencies
    from flask import request
    v_type = request.args.get("type", "ambulence")
    if v_type in ["ambulence", "fire_truck"]:
        simulated_emergencies.append(v_type)
        return jsonify({
            "status": "success",
            "message": f"{v_type} queued for simulation spawn"
        })
    return jsonify({
        "status": "error",
        "message": "Invalid emergency vehicle type"
    }), 400


@app.route("/set_spawn_rate")
def set_spawn_rate():
    global spawn_rate
    from flask import request
    rate_str = request.args.get("rate", "0.35")
    try:
        rate = float(rate_str)
        if 0.0 <= rate <= 1.0:
            spawn_rate = rate
            return jsonify({
                "status": "success",
                "spawn_rate": spawn_rate
            })
    except ValueError:
        pass
    return jsonify({
        "status": "error",
        "message": "Invalid spawn rate value"
    }), 400


@app.route("/set_cycle_time")
def set_cycle_time():
    global dqn_cycle_time
    from flask import request
    cycle_str = request.args.get("cycle", "5")
    try:
        cycle = int(cycle_str)
        if 2 <= cycle <= 30:
            dqn_cycle_time = cycle
            return jsonify({
                "status": "success",
                "dqn_cycle_time": dqn_cycle_time
            })
    except ValueError:
        pass
    return jsonify({
        "status": "error",
        "message": "Invalid cycle value. Must be between 2 and 30."
    }), 400


@app.route("/set_speed_limit")
def set_speed_limit():
    global simulation_speed_limit
    from flask import request
    speed_str = request.args.get("speed", "8")
    try:
        speed = int(speed_str)
        if 3 <= speed <= 15:
            simulation_speed_limit = speed
            return jsonify({
                "status": "success",
                "simulation_speed_limit": simulation_speed_limit
            })
    except ValueError:
        pass
    return jsonify({
        "status": "error",
        "message": "Invalid speed limit value. Must be between 3 and 15."
    }), 400


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True, port=5000)