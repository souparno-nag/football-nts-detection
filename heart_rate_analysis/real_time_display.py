class HeartRateMonitor:
    def __init__(self):
        self.data_history = {}
        
    def display_dashboard(self, player_data):
        """Simple console-based dashboard"""
        print("\n" + "="*50)
        print("HEART RATE MONITOR - LIVE DATA")
        print("="*50)
        
        for player in player_data:
            hr = player["heart_rate"]
            
            # Color coding for intensity
            if hr < 120:
                intensity = "🟢 LOW"
            elif hr < 160:
                intensity = "🟡 MODERATE"
            else:
                intensity = "🔴 HIGH"
                
            print(f"{player['player']:15} | {hr:3} BPM | {intensity:15} | {player['scenario']}")
        
        print("-"*50)