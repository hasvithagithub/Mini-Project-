# AURA Smart Traffic Hub 🚦

An intelligent, real-time Smart City Command Center designed to optimize urban traffic congestion and prioritize emergency response routes. Powered by a combination of computer vision tracking, LSTM forecasting, and Deep Q-Network (DQN) Reinforcement Learning agents.

---

## 🌟 Core Features

### 1. Multi-Mode Control Engine
Toggle between three advanced intersection control modes on-demand:
*   🤖 **PyTorch DQN Agent (Autonomous)**: Uses a Deep Q-Network model trained in a custom Gymnasium environment to dynamically switch traffic lights based on active queue balances and congestion rates.
*   ⏱️ **Fixed Timer Mode**: Periodic, automatic signal toggling using a standard countdown timer mapped directly to the cycle time configurations.
*   🎛️ **Manual Control**: Operator override to force either Lane A (North-South) or Lane B (East-West) green corridor state indefinitely.

### 2. Intelligent Computer Vision Pipelines
*   🚗 **Vehicle Detection & Tracking (YOLOv8)**: Real-time multi-class vehicle detection (cars, trucks, buses, motorcycles) and trajectory tracking across counting lines.
*   🚨 **EfficientNet Emergency Classification**: Crop-based classification that scans active vehicles for emergency profiles (Ambulance/Fire Truck) with high precision.
*   🛡️ **Instantaneous Emergency Corridor Override**: Immediately forces a green corridor for Lane A when an emergency vehicle is detected, reverting back to the normal cycle dynamically when clear.

### 3. Real-Time Analytics & Telemetry
*   📈 **LSTM Flow Forecasting**: Integrates a PyTorch LSTM time-series model to compare live vehicle density trends against predicted traffic volumes for the next hour.
*   📊 **Doughnut Composition Chart**: Dynamic visualization of active vehicle categories (Cars, Bikes, Buses, Trucks) on the road.
*   📜 **DQN Optimizer Console**: A scrollable terminal console directly on the dashboard reflecting decision logs, reward calculations, and phase history.
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

## 🧪 Simulation & Testing Guide

1.  Launch the dashboard and click **Start Stream** (defaulting to Simulated Traffic).
2.  Toggle to **Fixed Timer** and verify the active controller updates. Slide the cycle time slider to `10s` and watch the countdown trigger automatic light switches.
3.  Click **🚨 Spawn Ambulance** or **🔥 Spawn Fire Truck**. Notice that the red flashing emergency alert panel activates and forces Lane A green corridor *immediately*.
4.  Toggle to **Manual** and test forcing Lane B green.
5.  Observe the live **LSTM Traffic Density Forecasting** line chart and composition chart update dynamically as traffic flows.
