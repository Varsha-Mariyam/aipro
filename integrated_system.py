"""
Complete Aadhaar Verification System
OCR + Fraud Detection in one pipeline
"""

import cv2
import pytesseract
import re
import json
from fraud_detector import ForgeryDetector

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class AadhaarVerificationSystem:
    """
    Complete system: OCR → Validation → Fraud Detection
    Optimized for production use
    """
    
    def __init__(self):
        self.fraud_detector = ForgeryDetector()
    
    # ========================================
    # YOUR EXISTING OCR (OPTIMIZED)
    # ========================================
    
    def extract_text(self, img_path: str) -> str:
        """Your existing OCR code (optimized)"""
        img = cv2.imread(img_path)
        if img is None:
            return ""
        
        # Grayscale + preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 2)
        gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_LINEAR)
        
        # OCR
        text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
        text = text.replace("\n", " ").strip()
        
        return text
    
    def extract_fields(self, text: str) -> dict:
        """Your existing field extraction"""
        aadhaar = re.findall(r"\b\d{4}\s\d{4}\s\d{4}\b", text)
        dob = re.findall(r"(?:DOB|Date of Birth)[:\s\-]*([\d]{2}[./-][\d]{2}[./-][\d]{4})", text, re.I)
        name = re.findall(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})\b", text)
        
        # Filter names
        name = [n for n in name if not re.search(r"Government|India|Authority", n, re.I)]
        
        return {
            'name': name[0] if name else None,
            'dob': dob[0] if dob else None,
            'aadhaar': aadhaar[0] if aadhaar else None
        }
    
    def validate_fields(self, fields: dict) -> dict:
        """Field validation"""
        aadhaar_valid = bool(re.match(r"^\d{4}\s\d{4}\s\d{4}$", fields.get('aadhaar', '')))
        dob_valid = bool(re.match(r"^\d{2}[./-]\d{2}[./-]\d{4}$", fields.get('dob', '')))
        name_valid = fields.get('name') is not None
        
        return {
            'aadhaar_valid': aadhaar_valid,
            'dob_valid': dob_valid,
            'name_valid': name_valid,
            'all_valid': aadhaar_valid and dob_valid and name_valid
        }
    
    # ========================================
    # COMPLETE PIPELINE
    # ========================================
    
    def verify(self, img_path: str) -> dict:
        """
        Complete verification pipeline
        OCR → Validation → Fraud Detection → Final Decision
        """
        print(f"\n{'='*70}")
        print(f"🔹 COMPLETE AADHAAR VERIFICATION")
        print(f"{'='*70}\n")
        
        # Step 1: OCR
        print("1️⃣ Running OCR...")
        text = self.extract_text(img_path)
        
        if not text:
            return {'status': 'error', 'message': 'No text extracted'}
        
        # Step 2: Extract fields
        print("2️⃣ Extracting fields...")
        fields = self.extract_fields(text)
        print(f"   Name: {fields.get('name')}")
        print(f"   DOB: {fields.get('dob')}")
        print(f"   Aadhaar: {fields.get('aadhaar')}")
        
        # Step 3: Validate
        print("\n3️⃣ Validating fields...")
        validation = self.validate_fields(fields)
        
        # Step 4: Fraud detection
        print("\n4️⃣ Running fraud detection...")
        fraud_report = self.fraud_detector.analyze(img_path, verbose=True)
        
        # Step 5: Final decision
        print("\n5️⃣ Making final decision...")
        
        if not validation['all_valid']:
            final_decision = '🚫 REJECT - Invalid data'
        elif fraud_report['fraud_detected']:
            final_decision = fraud_report['recommendation']
        else:
            final_decision = '✅ APPROVE - All checks passed'
        
        # Compile complete report
        report = {
            'status': 'success',
            'final_decision': final_decision,
            'ocr': {
                'extracted_text': text,
                'fields': fields
            },
            'validation': validation,
            'fraud_detection': fraud_report
        }
        
        print(f"\n{'='*70}")
        print(f"📋 FINAL DECISION: {final_decision}")
        print(f"{'='*70}\n")
        
        return report
    
    def save_report(self, report: dict, output_file: str = 'verification_report.json'):
        """Save complete report"""
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Report saved: {output_file}")


# ========================================
# USAGE
# ========================================

if __name__ == "__main__":
    system = AadhaarVerificationSystem()
    
    # Test image
    test_img = r"C:\Users\varsh\OneDrive\Desktop\aiproject\Aadhaar\preprocessed_aadharcard_images\new_generated_aadharcard_images\1front_scaled_down.jpg"
    
    # Run complete verification
    report = system.verify(test_img)
    system.save_report(report)
