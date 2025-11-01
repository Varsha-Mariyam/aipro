"""
Test complete 2-layer fraud detection system
"""

from complete_system import CompleteFraudSystem
import os
import glob


def test_single_image():
    """Test on single image"""
    
    system = CompleteFraudSystem()
    
    test_img = r"C:\Users\varsh\OneDrive\Desktop\mmm\ai\Images\train\authentic\0c0584201ff552c4bdcbe160315aa432_jpg.rf.79c9a5cc723178ab7e79e869e7b4be75.jpg"
    
    report = system.analyze_complete(test_img, verbose=True)
    system.save_report(report, 'outputs/test_report.json')


def test_batch():
    """Test on multiple images"""
    
    system = CompleteFraudSystem()
    
    test_folder = r"C:\path\to\test\folder"
    images = glob.glob(f"{test_folder}/*.jpg")[:10]  # Test first 10
    
    results = []
    
    for i, img in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Testing: {os.path.basename(img)}")
        report = system.analyze_complete(img, verbose=False)
        results.append(report)
        print(f"   → {report['final_decision']['decision']}")
    
    # Summary
    print("\n" + "="*70)
    print("BATCH TEST SUMMARY")
    print("="*70)
    
    approved = sum(1 for r in results if 'APPROVE' in r['final_decision']['decision'])
    rejected = sum(1 for r in results if 'REJECT' in r['final_decision']['decision'])
    manual = sum(1 for r in results if 'MANUAL' in r['final_decision']['decision'])
    
    print(f"Total Tested:  {len(results)}")
    print(f"Approved:      {approved}")
    print(f"Rejected:      {rejected}")
    print(f"Manual Review: {manual}")
    print("="*70)


if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Single image test")
    print("2. Batch test")
    
    choice = input("\nEnter choice (1 or 2): ")
    
    if choice == '1':
        test_single_image()
    elif choice == '2':
        test_batch()
    else:
        print("Invalid choice!")
