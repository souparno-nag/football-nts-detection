class HeartRateAnalyzer:
    @staticmethod
    def calculate_variability(hr_readings):
        """Calculate Heart Rate Variability (HRV)"""
        if len(hr_readings) < 2:
            return 0
        
        differences = []
        for i in range(1, len(hr_readings)):
            differences.append(abs(hr_readings[i] - hr_readings[i-1]))
        
        return sum(differences) / len(differences) if differences else 0
    
    @staticmethod
    def detect_stress_patterns(hr_history):
        """Identify stress/arousal patterns"""
        if len(hr_history) < 5:
            return "Insufficient data"
        
        avg_hr = sum(hr_history) / len(hr_history)
        max_hr = max(hr_history)
        
        if max_hr > 180:
            return "⚠️  EXTREME EFFORT - Risk of overexertion"
        elif max_hr > 160 and avg_hr > 140:
            return "🔴 HIGH INTENSITY - Sustained pressure"
        elif avg_hr < 100:
            return "🟢 RECOVERY MODE - Low exertion"
        else:
            return "🟡 MODERATE INTENSITY - Optimal training"
    
    @staticmethod
    def zone_analysis(hr):
        """Calculate heart rate zones"""
        zones = {
            "Zone 1 (Recovery)": (50, 60),
            "Zone 2 (Aerobic)": (60, 70),
            "Zone 3 (Tempo)": (70, 80),
            "Zone 4 (Threshold)": (80, 90),
            "Zone 5 (Max)": (90, 100)
        }
        
        for zone, (low, high) in zones.items():
            if low <= (hr/200*100) <= high:  # Assuming max HR of 200
                return zone
        return "Unknown"