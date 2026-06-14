# AURA Smart Traffic Hub 🚦

An intelligent, real-time Smart City Command Center designed to optimize urban traffic congestion and prioritize emergency response routes. Powered by a combination of computer vision tracking, LSTM forecasting, and Deep Q-Network (DQN) Reinforcement Learning agents.

---

## 🌟 Core Features

### 1. Multi-Mode Control Engine (4-Way Chaurasta)
Toggle between three advanced intersection control modes designed for a 4-way chaurasta (North, South, East, West):
*   🤖 **PyTorch DQN Agent (Autonomous)**: Uses a Deep Q-Network model trained in a custom Gymnasium environment to dynamically switch traffic lights based on active queue balances and congestion rates across North-South vs East-West roads.
*   ⏱️ **Fixed Timer Mode**: Periodic, automatic signal toggling using a standard countdown timer mapped directly to the cycle time configurations.
*   🎛️ **Manual Control**: Operator override to force either Lane A (North-South) or Lane B (East-West) green corridor state indefinitely.

### 2. Intelligent Computer Vision Pipelines
*   🚗 **Vehicle Detection & Tracking (YOLOv8)**: Real-time multi-class vehicle detection (cars, trucks, buses, motorcycles) and trajectory tracking across counting lines.
*   🚨 **EfficientNet Emergency Classification**: Crop-based classification that scans active vehicles for emergency profiles (Ambulance/Fire Truck) with high precision.
*   🛡️ **Emergency Vehicle Bypass & Corridor Override**: Emergency vehicles (Ambulance/Fire Truck) completely bypass red lights and collision queue checks, driving through the chaurasta at full speed.

### 3. Real-Time Analytics & Telemetry
*   📈 **LSTM Flow Forecasting**: Integrates a PyTorch LSTM time-series model to compare live vehicle density trends against predicted traffic volumes for the next hour.
*   📊 **Doughnut Composition Chart**: Dynamic visualization of active vehicle categories (Cars, Bikes, Buses, Trucks) on the road.
*   📜 **DQN Optimizer Console**: A scrollable terminal console directly on the dashboard reflecting decision logs, reward calculations, and phase history.
*   📷 **CCTV Perspective Toggling**: Automatically switches camera view modes every 15 seconds between an **Aerial Top-down view** (orthographic) and an **Angled CCTV view** (perspective-warped corner projection).
*   🎚️ **Interactive Simulation Sliders**:
    *   **Simulator Congestion Level**: Adjusts traffic volume and spawn frequency (10% to 95%).
    *   **Signal Cycle Time**: Modifies DQN decision cycle intervals or Fixed Timer durations (2s to 25s).
    *   **Simulator Speed Limit**: Dynamically sets vehicle speed limits (30 km/h to 150 km/h) in the simulator.

---

## 🛠️ Tech Stack

*   **Backend**: Python, Flask, OpenCV
*   **Deep Learning & RL**: PyTorch, YOLOv8, EfficientNet, Gymnasium
*   **Frontend**: HTML5 (Semantic Structure), Vanilla CSS (Glassmorphic Dark Theme), JS, Chart.js (Served Locally)
*   **Data Science**: NumPy, Pandas, Scikit-learn

---

## 📦 Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/hasvithagithub/Mini-Project-.git
    cd Mini-Project-
    ```

2.  **Install Dependencies**:
    ```bash
    pip install flask opencv-python ultralytics numpy torch torchvision gymnasium
    ```

3.  **Run the Application**:
    ```bash
    python app.py
    ```

4.  **Open the Dashboard**:
    Open your browser and navigate to:
    *   Main Command Center: [http://127.0.0.1:5000](http://127.0.0.1:5000)
    *   Diagnostics Interface: [http://127.0.0.1:5000/camera](http://127.0.0.1:5000/camera)

---

## 🔬 Camera Stream Diagnostics Center

The upgraded Diagnostics Center (`/camera`) offers premium tools for calibrating and checking node statuses:
*   **CCTV Grid Overlay**: A toggleable visual crosshair and 3x3 alignment grid overlaid directly on the feed.
*   **Feed Adjustments**: Real-time adjustment sliders for **Feed Brightness**, **Feed Contrast**, and **Digital Zoom** applied via CSS filters and transforms.
*   **System Resources Telemetry**: Real-time animated progress bars tracking simulated **CPU Utilization**, **VRAM Allocation**, and **Network Bandwidth**.
*   **Node Diagnostics Console**: Monospace retro console with an interactive **Ping Test** runner that transmits packets to the local server, records individual latency, and compiles roundtrip statistics.

---

## 🧪 Simulation & Testing Guide

1.  Launch the dashboard and click **Start Stream** (defaulting to Simulated Traffic).
2.  Observe the 4-way intersection (chaurasta) graphics and vehicles flowing from North, South, East, and West directions.
3.  Watch the camera view mode automatically toggle between **Aerial Top-down** and **Angled CCTV** every 15 seconds.
4.  Toggle to **Fixed Timer** and verify the countdown counts down without stopping.
5.  Click **🚨 Spawn Ambulance** or **🔥 Spawn Fire Truck**. Notice that the emergency vehicle drives right through the red lights and other queues at full speed, while the countdown timer continues to cycle normally.
6.  Navigate to **Diagnostics** and test the Grid overlay, brightness sliders, and the interactive **Ping Test** console.

