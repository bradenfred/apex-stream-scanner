import json
import os
import time
import numpy as np
from collections import Counter

class LearningEngine:
    def __init__(self, data_file="training_data.json"):
        self.data_file = data_file
        self.learning_data = {
            "roi_history": {
                "kills": [],
                "squads": []
            },
            "visual_anchors": [],
            "text_patterns": {},
            "last_refined": 0
        }
        self.load_data()

    def load_data(self):
        """Load learning data from persistent storage"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    # Merge loaded data with default structure to ensure all keys exist
                    self.learning_data.update(data)
                print(f"🧠 Learning Engine: Loaded history with {len(self.learning_data['roi_history']['kills'])} samples")
            except Exception as e:
                print(f"⚠️ Learning Engine: Failed to load data: {e}")

    def save_data(self):
        """Save learning data to persistent storage"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.learning_data, f, indent=2)
        except Exception as e:
            print(f"❌ Learning Engine: Failed to save data: {e}")

    def record_roi_feedback(self, roi_type, roi_data):
        """Record a user-verified ROI for learning"""
        if roi_type not in self.learning_data["roi_history"]:
            return False
        
        # Add timestamp
        roi_data["timestamp"] = time.time()
        
        # Append to history
        self.learning_data["roi_history"][roi_type].append(roi_data)
        
        # Keep history manageable (last 100 entries)
        if len(self.learning_data["roi_history"][roi_type]) > 100:
            self.learning_data["roi_history"][roi_type].pop(0)
            
        self.save_data()
        
        # Trigger autonomous refinement if we have enough new data
        if time.time() - self.learning_data["last_refined"] > 300: # 5 minutes
            self.refine_models()
            
        return True

    def refine_models(self):
        """Autonomously refine detection models based on history"""
        print("🧠 Learning Engine: Refining models based on training data...")
        
        refined_rois = {}
        
        for roi_type in ["kills", "squads"]:
            history = self.learning_data["roi_history"][roi_type]
            if len(history) < 3:
                continue
                
            # Calculate optimal ROI based on median of verified positions
            # This ignores outliers (bad training data) automatically
            xs = [entry["x"] for entry in history]
            ys = [entry["y"] for entry in history]
            ws = [entry["width"] for entry in history]
            hs = [entry["height"] for entry in history]
            
            refined_rois[roi_type] = {
                "x": float(np.median(xs)),
                "y": float(np.median(ys)),
                "width": float(np.median(ws)),
                "height": float(np.median(hs))
            }
            
        if refined_rois:
            self.learning_data["refined_rois"] = refined_rois
            self.learning_data["last_refined"] = time.time()
            self.save_data()
            print(f"✨ Learning Engine: Models refined! New optimal ROIs calculated.")
            return refined_rois
            
        return None

    def get_optimal_roi(self, roi_type):
        """Get the current best ROI based on learning"""
        if "refined_rois" in self.learning_data and roi_type in self.learning_data["refined_rois"]:
            return self.learning_data["refined_rois"][roi_type]
        return None

# Global instance
engine = LearningEngine()
