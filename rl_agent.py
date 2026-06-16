# rl_agent.py
import torch
import torch.nn as nn
import numpy as np

# Define Q-Network (must match train_rl_agent.py structure)
class QNetwork(nn.Module):
    def __init__(self, state_dim=3, action_dim=2):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# Load the trained model
print("[RL Agent] Loading DQN controller weights from traffic_dqn.pt...")
try:
    model = QNetwork()
    model.load_state_dict(torch.load("traffic_dqn.pt"))
    model.eval()
    print("[RL Agent] DQN controller loaded successfully.")
except Exception as e:
    print(f"[RL Agent] Error loading weights: {e}. Falling back to default heuristics.")
    model = None

def get_rl_action(queue_a, queue_b, current_phase):
    """
    Inputs:
      - queue_a: integer queue length for lane A (N-S)
      - queue_b: integer queue length for lane B (E-W)
      - current_phase: 0 (Lane A green) or 1 (Lane B green)
    Returns:
      - action: 0 (Keep current phase) or 1 (Switch phase)
    """
    # 1. Fail-safe Heuristics Override
    # If the active lane is empty and the other lane has waiting vehicles, switch immediately
    if current_phase == 0 and queue_a == 0 and queue_b > 0:
        return 1
    if current_phase == 1 and queue_b == 0 and queue_a > 0:
        return 1
        
    # If the waiting lane is heavily congested and has significantly more vehicles than the green lane, switch
    if current_phase == 0 and queue_b >= 6 and queue_b > queue_a + 2:
        return 1
    if current_phase == 1 and queue_a >= 6 and queue_a > queue_b + 2:
        return 1

    # If the current green lane is heavily congested and the other lane is clear, keep it green
    if current_phase == 0 and queue_a >= 10 and queue_b < queue_a - 3:
        return 0
    if current_phase == 1 and queue_b >= 10 and queue_a < queue_b - 3:
        return 0

    # 2. DQN Model Prediction
    if model is not None:
        state = np.array([queue_a, queue_b, current_phase], dtype=np.float32)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            q_values = model(state_t)
            action = q_values.argmax().item()
            
        return action

    # 3. Fallback Heuristic if model loading failed and no override matched
    if current_phase == 0 and queue_b > queue_a:
        return 1
    if current_phase == 1 and queue_a > queue_b:
        return 1
    return 0

# Test helper
if __name__ == "__main__":
    test_cases = [
        (15, 2, 0),  # Phase 0, Lane A congested, Lane B clear -> action should be 0 (keep green on A)
        (2, 15, 0),  # Phase 0, Lane A clear, Lane B congested -> action should be 1 (switch to B)
        (2, 15, 1),  # Phase 1, Lane A clear, Lane B congested -> action should be 0 (keep green on B)
        (15, 2, 1),  # Phase 1, Lane A congested, Lane B clear -> action should be 1 (switch to A)
    ]
    for qa, qb, cp in test_cases:
        act = get_rl_action(qa, qb, cp)
        print(f"State: A={qa}, B={qb}, Phase={cp} | Recommended Action: {'SWITCH' if act == 1 else 'KEEP'}")
