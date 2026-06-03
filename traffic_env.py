# traffic_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class TrafficIntersectionEnv(gym.Env):
    """
    Custom Gymnasium Environment for a 2-Phase Intelligent Traffic Intersection.
    State representation: [Queue_Lane_A, Queue_Lane_B, Current_Green_Phase]
    Action space:
      - 0: Stay in the current phase
      - 1: Switch the traffic light phase
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, max_steps=200):
        super(TrafficIntersectionEnv, self).__init__()
        
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action space: Discrete(2) -> 0: Keep, 1: Switch
        self.action_space = spaces.Discrete(2)
        
        # Observation space: [Queue_A, Queue_B, Current_Phase]
        # Queue lengths capped at 50, phase is 0 or 1
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.int32),
            high=np.array([50, 50, 1], dtype=np.int32),
            dtype=np.int32
        )
        
        # Env parameters
        self.arrival_prob_a = 0.35  # Chance of vehicle arriving in Lane A
        self.arrival_prob_b = 0.25  # Chance of vehicle arriving in Lane B
        self.departure_rate = 2     # Vehicles cleared per step under green
        self.switch_penalty = 3.0   # Penalty for switching phases (simulates yellow delay)
        
        # State variables
        self.queue_a = 0
        self.queue_b = 0
        self.current_phase = 0       # 0: Lane A is green, 1: Lane B is green
        self.steps_in_phase = 0      # Track steps in current phase to prevent constant switching
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Random initial queues between 2 and 15
        self.queue_a = random.randint(2, 15)
        self.queue_b = random.randint(2, 15)
        self.current_phase = random.choice([0, 1])
        self.steps_in_phase = 0
        self.current_step = 0
        
        state = np.array([self.queue_a, self.queue_b, self.current_phase], dtype=np.int32)
        info = {}
        return state, info
        
    def step(self, action):
        self.current_step += 1
        self.steps_in_phase += 1
        
        # 1. Handle Phase Switch Action
        switched = False
        if action == 1:
            # Switch phase
            self.current_phase = 1 - self.current_phase
            self.steps_in_phase = 0
            switched = True
            
        # 2. Simulate Vehicle Arrivals (Poisson-like random arrivals)
        if random.random() < self.arrival_prob_a:
            self.queue_a = min(self.queue_a + 1, 50)
        if random.random() < self.arrival_prob_b:
            self.queue_b = min(self.queue_b + 1, 50)
            
        # 3. Simulate Vehicle Departures (Only for the active green phase)
        # Note: If we switched this step, we simulate a yellow phase delay (0 departures)
        departed = 0
        if not switched:
            if self.current_phase == 0:  # Lane A Green
                departed = min(self.queue_a, self.departure_rate)
                self.queue_a -= departed
            else:                        # Lane B Green
                departed = min(self.queue_b, self.departure_rate)
                self.queue_b -= departed
                
        # 4. Calculate State Reward
        # Reward is negative of queue lengths to encourage clearing the lanes
        reward = -1.0 * (self.queue_a + self.queue_b)
        
        # Penalty for switching phases (to prevent oscillation/constant switching)
        if switched:
            reward -= self.switch_penalty
            
        # Additional heavy penalty for excessive congestion (queue length > 15)
        if self.queue_a > 15:
            reward -= 2.0 * (self.queue_a - 15)
        if self.queue_b > 15:
            reward -= 2.0 * (self.queue_b - 15)
            
        # 5. Check Termination
        done = self.current_step >= self.max_steps
        truncated = False
        
        # State representation
        state = np.array([self.queue_a, self.queue_b, self.current_phase], dtype=np.int32)
        info = {
            "queue_a": self.queue_a,
            "queue_b": self.queue_b,
            "phase": self.current_phase,
            "departed": departed,
            "switched": switched
        }
        
        return state, reward, done, truncated, info
