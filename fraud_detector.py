import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import pytesseract
import os
from typing import Dict, Tuple
from dataclasses import dataclass
from enum import Enum

# Configuration
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


@dataclass
class FraudResult:
    """Structured fraud detection result"""
    technique: str
    detected: bool
    risk: RiskLevel
    confidence: float
    details: dict


class ForgeryDetector:
    """
    Optimized fraud detection with 4 techniques
    Single class, minimal footprint, fast execution
    """

    # Class-level constants
    ELA_HIGH = 600
    ELA_MEDIUM = 250
    FONT_VAR_THRESHOLD = 0.5  # Relaxed for real documents
    FONT_VAR_MEDIUM = 0.35
    EDGE_THRESHOLD = 0.15
    SUSPICIOUS_SOFTWARE = ['photoshop', 'gimp', 'paint.net', 'lightroom', 'affinity']

    def __init__(self):
        """Initialize with face cascade (load once)"""
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

   
    # TECHNIQUE 1: ERROR LEVEL ANALYSIS
   
    def detect_ela(self, img_path: str, quality: int = 90) -> FraudResult:
        """Error Level Analysis - Detects compression inconsistencies"""
        try:
            orig = Image.open(img_path).convert('RGB')
            temp_path = 'temp.jpg'
            orig.save(temp_path, 'JPEG', quality=quality)
            comp = Image.open(temp_path).convert('RGB')

            orig_arr = np.asarray(orig, dtype=np.float32)
            comp_arr = np.asarray(comp, dtype=np.float32)
            ela = np.abs(orig_arr - comp_arr)
            variance = float(np.var(ela))

            try:
                os.remove(temp_path)
            except:
                pass

            risk = (
                RiskLevel.HIGH if variance > self.ELA_HIGH
                else RiskLevel.MEDIUM if variance > self.ELA_MEDIUM
                else RiskLevel.LOW
            )

            return FraudResult(
                technique='ELA',
                detected=risk != RiskLevel.LOW,
                risk=risk,
                confidence=min(variance / self.ELA_HIGH, 1.0),
                details={'variance': variance}
            )

        except Exception as e:
            return FraudResult('ELA', False, RiskLevel.UNKNOWN, 0.0, {'error': str(e)})

   
    # TECHNIQUE 2: EXIF METADATA
   
    def detect_metadata(self, img_path: str) -> FraudResult:
        """EXIF metadata analysis"""
        try:
            img = Image.open(img_path)
            exif = img._getexif()

            if not exif:
                return FraudResult('EXIF', True, RiskLevel.MEDIUM, 0.5,
                                   {'reason': 'No EXIF (stripped)'})

            software = str(exif.get(305, '')).lower()
            for tool in self.SUSPICIOUS_SOFTWARE:
                if tool in software:
                    return FraudResult('EXIF', True, RiskLevel.HIGH, 0.9,
                                       {'software': software})

            return FraudResult('EXIF', False, RiskLevel.LOW, 0.1,
                               {'software': software or 'unknown'})

        except Exception as e:
            return FraudResult('EXIF', False, RiskLevel.UNKNOWN, 0.0, {'error': str(e)})

   
    # TECHNIQUE 3: PHOTO OVERLAY DETECTION
   
    def detect_overlay(self, img_path: str) -> FraudResult:
        """Photo overlay detection using edge analysis"""
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            edges = cv2.Canny(img, 100, 200)
            faces = self._face_cascade.detectMultiScale(img, 1.1, 4)

            if len(faces) == 0:
                return FraudResult('OVERLAY', False, RiskLevel.UNKNOWN, 0.0,
                                   {'reason': 'No face detected'})

            x, y, w, h = faces[0]
            y1, y2 = max(0, y - 10), min(img.shape[0], y + h + 10)
            x1, x2 = max(0, x - 10), min(img.shape[1], x + w + 10)
            boundary = edges[y1:y2, x1:x2]

            density = np.count_nonzero(boundary) / boundary.size
            risk = (
                RiskLevel.HIGH if density > self.EDGE_THRESHOLD
                else RiskLevel.MEDIUM if density > self.EDGE_THRESHOLD * 0.6
                else RiskLevel.LOW
            )

            return FraudResult('OVERLAY', risk != RiskLevel.LOW, risk,
                               min(density / self.EDGE_THRESHOLD, 1.0),
                               {'edge_density': float(density)})

        except Exception as e:
            return FraudResult('OVERLAY', False, RiskLevel.UNKNOWN, 0.0, {'error': str(e)})

    
    # TECHNIQUE 4: FONT CONSISTENCY
   
    def detect_fonts(self, img_path: str) -> FraudResult:
        """Font consistency using OCR confidence"""
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            data = pytesseract.image_to_data(
                img, output_type=pytesseract.Output.DICT, config='--psm 6 --oem 3'
            )

            confs = np.array([float(c) for c in data['conf'] if c != '-1' and float(c) > 0])

            if len(confs) < 3:
                return FraudResult('FONT', False, RiskLevel.UNKNOWN, 0.0,
                                   {'reason': 'Insufficient text'})

            conf_range = (confs.max() - confs.min()) / 100

            # 3-tier threshold
            if conf_range > self.FONT_VAR_THRESHOLD:
                risk = RiskLevel.HIGH
            elif conf_range > self.FONT_VAR_MEDIUM:
                risk = RiskLevel.MEDIUM
            else:
                risk = RiskLevel.LOW

            return FraudResult('FONT', risk != RiskLevel.LOW, risk,
                               min(conf_range / self.FONT_VAR_THRESHOLD, 1.0),
                               {'conf_range': float(conf_range), 'avg_conf': float(confs.mean())})

        except Exception as e:
            return FraudResult('FONT', False, RiskLevel.UNKNOWN, 0.0, {'error': str(e)})

   
    # COMPREHENSIVE ANALYSIS
   
    def analyze(self, img_path: str, verbose: bool = True) -> Dict:
        """Run all fraud checks"""
        if verbose:
            print(f"\n{'='*70}\n🔍 Analyzing: {os.path.basename(img_path)}\n{'='*70}")

        results = {
            'ela': self.detect_ela(img_path),
            'metadata': self.detect_metadata(img_path),
            'overlay': self.detect_overlay(img_path),
            'fonts': self.detect_fonts(img_path)
        }

        risks = [r.risk for r in results.values()]
        risk_counts = {
            RiskLevel.HIGH: risks.count(RiskLevel.HIGH),
            RiskLevel.MEDIUM: risks.count(RiskLevel.MEDIUM),
            RiskLevel.LOW: risks.count(RiskLevel.LOW)
        }

        if risk_counts[RiskLevel.HIGH] >= 1:
            overall_risk, fraud, recommendation = RiskLevel.HIGH, True, '🚫 REJECT'
        elif risk_counts[RiskLevel.HIGH] + risk_counts[RiskLevel.MEDIUM] >= 2:
            overall_risk, fraud, recommendation = RiskLevel.HIGH, True, '🚫 REJECT'
        elif risk_counts[RiskLevel.MEDIUM] >= 1:
            overall_risk, fraud, recommendation = RiskLevel.MEDIUM, True, '⚠️ MANUAL_REVIEW'
        else:
            overall_risk, fraud, recommendation = RiskLevel.LOW, False, '✅ APPROVE'

        report = {
            'image': img_path,
            'fraud_detected': fraud,
            'overall_risk': overall_risk.value,
            'recommendation': recommendation,
            'risk_counts': {k.value: v for k, v in risk_counts.items()},
            'techniques': {
                name: {
                    'detected': res.detected,
                    'risk': res.risk.value,
                    'confidence': res.confidence,
                    'details': res.details
                } for name, res in results.items()
            }
        }

        if verbose:
            self._print_report(report)
        return report

    def _print_report(self, report: Dict):
        """Pretty print report"""
        print(f"\n📊 FRAUD DETECTION REPORT")
        print(f"{'='*70}")
        print(f"Overall Risk:      {report['overall_risk']}")
        print(f"Fraud Detected:    {'YES' if report['fraud_detected'] else 'NO'}")
        print(f"Recommendation:    {report['recommendation']}")
        print(f"\nRisk Breakdown:")
        for tech, data in report['techniques'].items():
            print(f"  {tech.upper():12} → {data['risk']:8} (conf: {data['confidence']:.2f})")
        print(f"{'='*70}\n")


# BATCH PROCESSOR
class BatchProcessor:
    """Process multiple images efficiently"""

    def __init__(self):
        self.detector = ForgeryDetector()

    def process_folder(self, folder_path: str, output_file: str = 'results.json'):
        import glob
        import json

        patterns = ['*.jpg', '*.jpeg', '*.png']
        images = []
        for pattern in patterns:
            images.extend(glob.glob(os.path.join(folder_path, pattern)))

        print(f"📁 Found {len(images)} images")

        results = []
        for i, img in enumerate(images, 1):
            print(f"\n[{i}/{len(images)}] Processing {os.path.basename(img)}")
            try:
                result = self.detector.analyze(img, verbose=False)
                results.append(result)
                print(f"   → {result['recommendation']}")
            except Exception as e:
                print(f"   ❌ Error: {e}")

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}")


# MAIN EXECUTION
if __name__ == "__main__":
    detector = ForgeryDetector()
    test_img = r"C:\Users\varsh\OneDrive\Desktop\mmm\ai\Images\train\authentic\0c0584201ff552c4bdcbe160315aa432_jpg.rf.3146b68fa30c1a246288d8373be2f2d8.jpg"
    detector.analyze(test_img)
