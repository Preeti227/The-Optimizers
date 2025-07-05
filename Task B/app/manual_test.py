import os
import sys

# Add the *parent* directory of "app" to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.manual_verification import verify_identity_pair

def run_manual_verification_cli():
    print("\nManual Identity Verification")

    ref_path = input("Enter full path to the reference image: ").strip()
    test_path = input("Enter full path to the test image (possibly distorted): ").strip()
    distortion = input("Enter distortion type (or leave blank for auto): ").strip().lower()
    threshold_input = input("Enter similarity threshold (default = 0.5): ").strip()

    threshold = float(threshold_input) if threshold_input else 0.5

    verify_identity_pair(
        ref_img_path=ref_path,
        test_img_path=test_path,
        distortion_type=distortion if distortion else None,
        threshold=threshold
    )

if __name__ == "__main__":
    run_manual_verification_cli()
