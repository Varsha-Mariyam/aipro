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
        print("🔄 Initializing Complete Fraud Detection System...")
        print("=" * 70)

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
        print("=" * 70)
        print("✅ System initialized successfully!\n")

    # ---------------------------------------------------------
    # Preprocess image for CNN
    # ---------------------------------------------------------
    def preprocess_for_cnn(self, image_path):
        img = keras.preprocessing.image.load_img(image_path, target_size=self.img_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    # ---------------------------------------------------------
    # CNN Prediction
    # ---------------------------------------------------------
    def cnn_predict(self, image_path):
        if self.cnn_model is None:
            return {
                'available': False,
                'fake_probability': 0.5,
                'confidence': 0.0,
                'prediction': 'UNKNOWN',
                'error': 'CNN model not loaded'
            }

        try:
            img_array = self.preprocess_for_cnn(image_path)
            prediction = self.cnn_model.predict(img_array, verbose=0)[0][0]
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

    # ---------------------------------------------------------
    # Complete Analysis (Layer 1 + Layer 2)
    # ---------------------------------------------------------
    def analyze_complete(self, image_path, verbose=True):
        if verbose:
            print(f"\n{'='*70}")
            print("🔍 COMPLETE FRAUD ANALYSIS")
            print(f"{'='*70}")
            print(f"Image: {os.path.basename(image_path)}\n")

        # LAYER 1
        if verbose:
            print("🔹 LAYER 1: Algorithmic Detection")
            print("-" * 70)
        day1_result = self.algo_detector.analyze(image_path, verbose=False)

        if verbose:
            print(f"   ELA:      {day1_result['techniques']['ela']['risk']}")
            print(f"   Metadata: {day1_result['techniques']['metadata']['risk']}")
            print(f"   Overlay:  {day1_result['techniques']['overlay']['risk']}")
            print(f"   Fonts:    {day1_result['techniques']['fonts']['risk']}")
            print(f"   Overall:  {day1_result['overall_risk']}\n")

        # LAYER 2
        if verbose:
            print("🔹 LAYER 2: CNN Deep Learning")
            print("-" * 70)
        day2_result = self.cnn_predict(image_path)

        if verbose:
            if day2_result['available']:
                print(f"   Prediction:     {day2_result['prediction']}")
                print(f"   Fake Prob:      {day2_result['fake_probability']:.2%}")
                print(f"   Authentic Prob: {day2_result['authentic_probability']:.2%}")
                print(f"   Confidence:     {day2_result['confidence']:.2%}\n")
            else:
                print(f"   ⚠️ CNN not available: {day2_result.get('error', 'Unknown')}\n")

        final_decision = self.make_final_decision(day1_result, day2_result)

        if verbose:
            print("🔹 FINAL DECISION")
            print("-" * 70)
            print(f"   Decision:   {final_decision['decision']}")
            print(f"   Confidence: {final_decision['confidence']:.2%}")
            print(f"   Reasoning:  {final_decision['reasoning']}")
            print("=" * 70 + "\n")

        report = {
            'image': image_path,
            'layer1_algorithmic': day1_result,
            'layer2_cnn': day2_result,
            'final_decision': final_decision
        }

        return report

    # ---------------------------------------------------------
    # Final Decision Logic (Properly Indented)
    # ---------------------------------------------------------
    def make_final_decision(self, day1, day2):
        day1_risk = day1['overall_risk']
        day2_available = day2['available']

        if not day2_available:
            if day1_risk == 'HIGH':
                return {
                    'decision': '🚫 REJECT',
                    'confidence': 0.70,
                    'reasoning': 'High algorithmic risk (CNN unavailable)'
                }
            elif day1_risk == 'MEDIUM':
                return {
                    'decision': '⚠ MANUAL_REVIEW',
                    'confidence': 0.50,
                    'reasoning': 'Medium algorithmic risk (CNN unavailable)'
                }
            else:
                return {
                    'decision': '✅ APPROVE',
                    'confidence': 0.65,
                    'reasoning': 'Low algorithmic risk (CNN unavailable)'
                }

        day2_fake_prob = day2['fake_probability']
        day2_confidence = day2['confidence']
        risk_counts = day1.get('risk_counts', {})
        high_count = risk_counts.get('HIGH', 0)
        medium_count = risk_counts.get('MEDIUM', 0)

        # Case 1: CNN is very confident
        if day2_confidence > 0.95:
            if day2_fake_prob < 0.05:
                if day1_risk == 'HIGH' and high_count >= 2:
                    return {
                        'decision': '⚠ MANUAL_REVIEW',
                        'confidence': 0.75,
                        'reasoning': 'CNN confident authentic, but multiple Day 1 risks'
                    }
                else:
                    return {
                        'decision': '✅ APPROVE',
                        'confidence': 0.92,
                        'reasoning': 'CNN highly confident document is authentic'
                    }

            elif day2_fake_prob > 0.95:
                return {
                    'decision': '🚫 REJECT',
                    'confidence': 0.95,
                    'reasoning': 'CNN highly confident document is fake'
                }

        # Case 2: Both agree
        if day1_risk == 'HIGH' and day2_fake_prob > 0.7:
            return {
                'decision': '🚫 REJECT',
                'confidence': 0.95,
                'reasoning': 'Both layers detected high fraud risk'
            }

        if day1_risk == 'LOW' and day2_fake_prob < 0.3:
            return {
                'decision': '✅ APPROVE',
                'confidence': 0.95,
                'reasoning': 'Both layers confirm authenticity'
            }

        # Case 3: Conflicting signals
        if day1_risk == 'HIGH' and day2_fake_prob < 0.3:
            if high_count == 1:
                return {
                    'decision': '✅ APPROVE',
                    'confidence': 0.80,
                    'reasoning': f'CNN confident authentic (fake prob: {day2_fake_prob:.1%}), only one Day 1 risk'
                }
            else:
                return {
                    'decision': '⚠ MANUAL_REVIEW',
                    'confidence': 0.60,
                    'reasoning': 'CNN says authentic but multiple Day 1 risks detected'
                }

        if day1_risk == 'LOW' and day2_fake_prob > 0.7:
            return {
                'decision': '🚫 REJECT',
                'confidence': 0.85,
                'reasoning': 'CNN detected fraud patterns despite clean Day 1 checks'
            }

        # Case 4: Medium risks
        if day1_risk == 'MEDIUM' or (0.3 <= day2_fake_prob <= 0.7):
            return {
                'decision': '⚠ MANUAL_REVIEW',
                'confidence': 0.60,
                'reasoning': 'Uncertain indicators - requires human verification'
            }

        # Case 5: Default
        return {
            'decision': '✅ APPROVE',
            'confidence': 0.75,
            'reasoning': 'Low fraud risk overall'
        }

    # ---------------------------------------------------------
    # Save JSON Report
    # ---------------------------------------------------------
    def save_report(self, report, output_file='outputs/complete_report.json'):
        import json
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"✅ Report saved: {output_file}")


# ---------------------------------------------------------
# Run script directly
# ---------------------------------------------------------
if __name__ == "__main__":
    system = CompleteFraudSystem(
        cnn_model_path='models/fraud_cnn.h5',
        img_size=(128, 128)
    )

    test_image = r"C:\Users\varsh\OneDrive\Desktop\mmm\ai\Images\train\authentic\0c0584201ff552c4bdcbe160315aa432_jpg.rf.3146b68fa30c1a246288d8373be2f2d8.jpg"
    report = system.analyze_complete(test_image, verbose=True)
    system.save_report(report)
