"""
Complete Fraud Detection System
Combines Day 1 (Algorithmic) + Day 2 (CNN) approaches
"""

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os

# Import Day 1 fraud detector
from fraud_detector import ForgeryDetector


class CompleteFraudSystem:
    """
    2-Layer Fraud Detection System
    Layer 1: Algorithmic (ELA, EXIF, Overlay, Fonts)
    Layer 2: Deep Learning (CNN pattern recognition)
    """
    
    def __init__(self, cnn_model_path='models/fraud_cnn.h5', img_size=(128, 128)):
        """
        Initialize complete system
        
        Args:
            cnn_model_path: Path to trained CNN model
            img_size: Image size for CNN (must match training size)
        """
        
        print("🔄 Initializing Complete Fraud Detection System...")
        print("="*70)
        
        # Layer 1: Algorithmic detection
        print("📦 Loading Day 1 (Algorithmic) detector...")
        self.algo_detector = ForgeryDetector()
        
        # Layer 2: CNN detection
        print("🧠 Loading Day 2 (CNN) model...")
        if os.path.exists(cnn_model_path):
            self.cnn_model = keras.models.load_model(cnn_model_path)
            print(f"   ✅ CNN model loaded from: {cnn_model_path}")
        else:
            print(f"   ⚠️ CNN model not found: {cnn_model_path}")
            print("   Run training first: python cnn_model.py")
            self.cnn_model = None
        
        self.img_size = img_size
        
        print("="*70)
        print("✅ System initialized successfully!\n")
    
    def preprocess_for_cnn(self, image_path):
        """
        Preprocess image for CNN prediction
        
        Args:
            image_path: Path to image
        
        Returns:
            Preprocessed image array
        """
        # Load and resize image
        img = keras.preprocessing.image.load_img(
            image_path,
            target_size=self.img_size
        )
        
        # Convert to array and normalize
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def cnn_predict(self, image_path):
        """
        Get CNN prediction for image
        
        Returns:
            Dictionary with prediction results
        """
        if self.cnn_model is None:
            return {
                'available': False,
                'fake_probability': 0.5,
                'confidence': 0.0,
                'prediction': 'UNKNOWN',
                'error': 'CNN model not loaded'
            }
        
        try:
            # Preprocess image
            img_array = self.preprocess_for_cnn(image_path)
            
            # Get prediction
            prediction = self.cnn_model.predict(img_array, verbose=0)[0][0]
            
            # Interpret prediction
            # Output is 0-1: 0=authentic, 1=fake
            fake_prob = float(prediction)
            
            if fake_prob > 0.7:
                pred_class = 'FAKE'
                confidence = fake_prob
            elif fake_prob < 0.3:
                pred_class = 'AUTHENTIC'
                confidence = 1 - fake_prob
            else:
                pred_class = 'UNCERTAIN'
                confidence = 0.5
            
            return {
                'available': True,
                'fake_probability': fake_prob,
                'authentic_probability': 1 - fake_prob,
                'confidence': confidence,
                'prediction': pred_class
            }
        
        except Exception as e:
            return {
                'available': False,
                'fake_probability': 0.5,
                'confidence': 0.0,
                'prediction': 'ERROR',
                'error': str(e)
            }
    
    def analyze_complete(self, image_path, verbose=True):
        """
        Run complete 2-layer fraud analysis
        
        Args:
            image_path: Path to Aadhaar image
            verbose: Print detailed output
        
        Returns:
            Complete fraud analysis report
        """
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"🔍 COMPLETE FRAUD ANALYSIS")
            print(f"{'='*70}")
            print(f"Image: {os.path.basename(image_path)}\n")
        
        # LAYER 1: Algorithmic Detection
        if verbose:
            print("🔹 LAYER 1: Algorithmic Detection")
            print("-"*70)
        
        day1_result = self.algo_detector.analyze(image_path, verbose=False)
        
        if verbose:
            print(f"   ELA:      {day1_result['techniques']['ela']['risk']}")
            print(f"   Metadata: {day1_result['techniques']['metadata']['risk']}")
            print(f"   Overlay:  {day1_result['techniques']['overlay']['risk']}")
            print(f"   Fonts:    {day1_result['techniques']['fonts']['risk']}")
            print(f"   Overall:  {day1_result['overall_risk']}\n")
        
        # LAYER 2: CNN Detection
        if verbose:
            print("🔹 LAYER 2: CNN Deep Learning")
            print("-"*70)
        
        day2_result = self.cnn_predict(image_path)
        
        if verbose:
            if day2_result['available']:
                print(f"   Prediction:    {day2_result['prediction']}")
                print(f"   Fake Prob:     {day2_result['fake_probability']:.2%}")
                print(f"   Authentic Prob: {day2_result['authentic_probability']:.2%}")
                print(f"   Confidence:    {day2_result['confidence']:.2%}\n")
            else:
                print(f"   ⚠️ CNN not available: {day2_result.get('error', 'Unknown')}\n")
        
        # COMBINED DECISION
        final_decision = self.make_final_decision(day1_result, day2_result)
        
        if verbose:
            print("🔹 FINAL DECISION")
            print("-"*70)
            print(f"   Decision: {final_decision['decision']}")
            print(f"   Confidence: {final_decision['confidence']:.2%}")
            print(f"   Reasoning: {final_decision['reasoning']}")
            print("="*70 + "\n")
        
        # Compile complete report
        report = {
            'image': image_path,
            'layer1_algorithmic': day1_result,
            'layer2_cnn': day2_result,
            'final_decision': final_decision
        }
        
        return report
    
    def make_final_decision(self, day1, day2):
        """
        Combine Day 1 and Day 2 results for final decision
        
        Decision Logic:
        - If both HIGH risk → REJECT (high confidence)
        - If both LOW risk → APPROVE (high confidence)
        - If conflicting → MANUAL_REVIEW (low confidence)
        """
        
        day1_risk = day1['overall_risk']
        day2_available = day2['available']
        
        if not day2_available:
            # CNN not available, use only Day 1
            if day1_risk == 'HIGH':
                return {
                    'decision': '🚫 REJECT',
                    'confidence': 0.70,
                    'reasoning': 'High algorithmic risk (CNN not available)'
                }
            elif day1_risk == 'MEDIUM':
                return {
                    'decision': '⚠️ MANUAL_REVIEW',
                    'confidence': 0.50,
                    'reasoning': 'Medium algorithmic risk (CNN not available)'
                }
            else:
                return {
                    'decision': '✅ APPROVE',
                    'confidence': 0.60,
                    'reasoning': 'Low algorithmic risk (CNN not available)'
                }
        
        # Both layers available
        day2_fake_prob = day2['fake_probability']
        
        # High confidence REJECT (both agree it's fake)
        if day1_risk == 'HIGH' and day2_fake_prob > 0.7:
            return {
                'decision': '🚫 REJECT',
                'confidence': 0.95,
                'reasoning': 'Both layers detected high fraud risk'
            }
        
        # Medium-high REJECT (one strong signal)
        if day1_risk == 'HIGH' or day2_fake_prob > 0.8:
            return {
                'decision': '🚫 REJECT',
                'confidence': 0.85,
                'reasoning': 'Strong fraud indicator from one layer'
            }
        
        # MANUAL_REVIEW (conflicting or medium signals)
        if (day1_risk == 'HIGH' and day2_fake_prob < 0.4) or \
           (day1_risk == 'LOW' and day2_fake_prob > 0.6):
            return {
                'decision': '⚠️ MANUAL_REVIEW',
                'confidence': 0.50,
                'reasoning': 'Conflicting signals between layers'
            }
        
        if day1_risk == 'MEDIUM' or (0.4 <= day2_fake_prob <= 0.6):
            return {
                'decision': '⚠️ MANUAL_REVIEW',
                'confidence': 0.60,
                'reasoning': 'Medium risk indicators detected'
            }
        
        # High confidence APPROVE (both agree it's authentic)
        if day1_risk == 'LOW' and day2_fake_prob < 0.3:
            return {
                'decision': '✅ APPROVE',
                'confidence': 0.95,
                'reasoning': 'Both layers confirm authenticity'
            }
        
        # Default APPROVE (mostly low risk)
        return {
            'decision': '✅ APPROVE',
            'confidence': 0.80,
            'reasoning': 'Low fraud risk overall'
        }
    
    def save_report(self, report, output_file='outputs/complete_report.json'):
        """Save complete report to JSON"""
        import json
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Report saved: {output_file}")


# Test the complete system
if __name__ == "__main__":
    # Initialize system
    system = CompleteFraudSystem(
        cnn_model_path='models/fraud_cnn.h5',
        img_size=(128, 128)
    )
    
    # Test image
    test_image = r"C:\Users\varsh\OneDrive\Desktop\mmm\ai\Images\train\authentic\2bfd2f150b31581bac34445b5c49dd26_jpg.rf.286e61ad0990d819c133d79bf680ab10.jpg"
    
    # Run complete analysis
    report = system.analyze_complete(test_image, verbose=True)
    
    # Save report
    system.save_report(report)
