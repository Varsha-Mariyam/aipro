"""
Test suite for fraud detection system
"""

from integrated_system import AadhaarVerificationSystem
import glob
import time


def test_single():
    """Test single image"""
    system = AadhaarVerificationSystem()
    
    test_img = r"C:\Users\varsh\OneDrive\Desktop\aiproject\Aadhaar\preprocessed_aadharcard_images\new_generated_aadharcard_images\1front_scaled_down.jpg"
    start = time.time()
    report = system.verify(test_img)
    duration = time.time() - start
    
    print(f"\n⏱️ Processing time: {duration:.2f} seconds")
    
    return report


def test_batch():
    """Test multiple images"""
    system = AadhaarVerificationSystem()
    
    folder = r"C:\Users\varsh\OneDrive\Desktop\aiproject\Aadhaar\preprocessed_aadharcard_images\new_generated_aadharcard_images"
    images = glob.glob(f"{folder}/*.jpg")
    
    results = []
    total_time = 0
    
    for img in images[:10]:  # Test first 10
        start = time.time()
        result = system.verify(img)
        duration = time.time() - start
        
        total_time += duration
        results.append(result)
        
        print(f"Processed in {duration:.2f}s")
    
    avg_time = total_time / len(results)
    print(f"\n⏱️ Average time: {avg_time:.2f} seconds/image")
    
    return results


def benchmark():
    """Performance benchmark"""
    from fraud_detector import ForgeryDetector
    
    detector = ForgeryDetector()
    test_img = r"C:\Users\varsh\OneDrive\Desktop\aiproject\Aadhaar\preprocessed_aadharcard_images\new_generated_aadharcard_images\1front_scaled_down.jpg"
    
    # Test individual techniques
    techniques = [
        ('ELA', lambda: detector.detect_ela(test_img)),
        ('EXIF', lambda: detector.detect_metadata(test_img)),
        ('Overlay', lambda: detector.detect_overlay(test_img)),
        ('Font', lambda: detector.detect_fonts(test_img))
    ]
    
    print("\n⏱️ PERFORMANCE BENCHMARK")
    print("="*50)
    
    for name, func in techniques:
        times = []
        for _ in range(5):  # 5 runs
            start = time.time()
            func()
            times.append(time.time() - start)
        
        avg = sum(times) / len(times) * 1000  # Convert to ms
        print(f"{name:12} → {avg:6.1f} ms")
    
    print("="*50)


if __name__ == "__main__":
    # Run tests
    print("Testing single image...")
    test_single()
    
    print("\n\nTesting batch...")
    test_batch()
    
    print("\n\nBenchmarking...")
    benchmark()
