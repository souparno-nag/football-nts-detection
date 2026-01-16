from data_gen import DummyHeartRateGenerator
from real_time_display import HeartRateMonitor
from data_analysis import HeartRateAnalyzer
import random
import time

class FootballHeartRateSimulator:
    def __init__(self):
        self.players = ["Player_1", "Player_2", "Player_3", "Player_4", "Player_5"]
        self.generator = DummyHeartRateGenerator(self.players)
        self.monitor = HeartRateMonitor()
        self.analyzer = HeartRateAnalyzer()
        self.history = {player: [] for player in self.players}
        
    def simulate_match_scenario(self, duration_minutes=5):
        """Simulate a match with different scenarios"""
        scenarios = ["normal", "sprint", "pressure", "recovery", "critical_moment"]
        
        print("🚀 Starting Heart Rate Simulation...")
        print(f"📊 Monitoring {len(self.players)} players")
        print(f"⏱️  Duration: {duration_minutes} minutes")
        
        for minute in range(duration_minutes):
            print(f"\n⏰ MINUTE {minute + 1}")
            
            current_readings = []
            
            for player in self.players:
                # Random scenario for each player
                scenario = random.choice(scenarios)
                if minute == 0:
                    scenario = "normal"
                elif minute == duration_minutes - 1:
                    scenario = "recovery"
                
                data = self.generator.generate_reading(player, scenario)
                self.history[player].append(data["heart_rate"])
                current_readings.append(data)
            
            # Display dashboard
            self.monitor.display_dashboard(current_readings)
            
            # Show analysis for first player as example
            if minute > 0:
                self._show_sample_analysis()
            
            time.sleep(2)  # Simulate real-time delay
    
    def _show_sample_analysis(self):
        """Display analysis for a sample player"""
        sample_player = self.players[0]
        hr_history = self.history[sample_player]
        
        if len(hr_history) > 1:
            print(f"\n📈 ANALYSIS for {sample_player}:")
            print(f"   • Current HR: {hr_history[-1]} BPM")
            print(f"   • HR Zone: {self.analyzer.zone_analysis(hr_history[-1])}")
            print(f"   • HR Variability: {self.analyzer.calculate_variability(hr_history[-5:]):.1f}")
            print(f"   • Pattern: {self.analyzer.detect_stress_patterns(hr_history[-5:])}")
    
    def generate_report(self):
        """Generate end-of-session report"""
        print("\n" + "="*60)
        print("SESSION ANALYSIS REPORT")
        print("="*60)
        
        for player in self.players:
            if self.history[player]:
                avg_hr = sum(self.history[player]) / len(self.history[player])
                max_hr = max(self.history[player])
                min_hr = min(self.history[player])
                
                print(f"\n{player}:")
                print(f"  📊 Avg HR: {avg_hr:.0f} BPM")
                print(f"  📈 Max HR: {max_hr} BPM")
                print(f"  📉 Min HR: {min_hr} BPM")
                print(f"  🔄 HR Variability: {self.analyzer.calculate_variability(self.history[player]):.1f}")
                
                # Performance assessment
                if avg_hr > 140:
                    print(f"  💪 Performance: HIGH INTENSITY session")
                elif avg_hr > 115:
                    print(f"  ✅ Performance: OPTIMAL training load")
                else:
                    print(f"  🧘 Performance: RECOVERY focus")

# Main execution
if __name__ == "__main__":
    simulator = FootballHeartRateSimulator()
    simulator.simulate_match_scenario(duration_minutes=3)
    simulator.generate_report()