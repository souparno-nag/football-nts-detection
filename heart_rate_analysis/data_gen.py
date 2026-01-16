import random
from datetime import datetime

class DummyHeartRateGenerator:
    def __init__(self, player_names, baseline_hr:int=120):
        self.players = {name: baseline_hr for name in player_names}
        self.activity_states = {name: "resting" for name in player_names}

    def generate_reading(self, player_name, scenario="normal"):
        """Generate realistic heart rate based on scenario"""
        base = self.players[player_name]

        # Scenario based modifiers
        modifiers = {
            "normal": random.randint(-3, 3),
            "sprint": random.randint(30, 50),
            "pressure": random.randint(20, 35),
            "recovery": random.randint(-10, -5),
            "critical": random.randint(40, 60)
        }

        hr = base + modifiers.get(scenario, 0)
        hr = max(55, min(hr, 200))

        return {
            "player": player_name,
            "heart_rate": hr,
            "timestamp": datetime.now().isoformat(),
            "scenario": scenario
        }