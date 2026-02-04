import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detection.settings')
import django
django.setup()

from detector.ml_utils import get_detector  # ✅ FIXED: Use get_detector() singleton
from django.conf import settings
import numpy as np
import cv2

print("\n" + "="*70)
print("🧪 IMAGE DEEPFAKE DETECTION TEST - FIXED VERSION")
print("="*70)

print(f"\n⚙️  Configuration:")
print(f"   Model Type: {settings.MODEL_TYPE}")
print(f"   Image Size: {settings.IMAGE_SIZE}x{settings.IMAGE_SIZE}")
print(f"   Model Path: {settings.MODEL_PATH}")
print(f"   Model Exists: {settings.MODEL_PATH.exists()}")

if not settings.MODEL_PATH.exists():
    print(f"\n❌ ERROR: Model not found!")
    print(f"\n   Please train the model first:")
    print(f"   python training_scripts/train_efficientnet_b1.py")
    exit(1)

# ✅ FIXED: Get detector instance
print("\n📥 Loading detector...")
detector = get_detector()

# Get model info
print(f"\n🔍 Model Information:")
model_info = detector.get_model_info()
print(f"   Input size: {model_info['input_size']}x{model_info['input_size']}")
print(f"   Output type: {model_info['output_type']}")
print(f"   Total parameters: {model_info['total_parameters']:,}")
print(f"   Model output shape: {detector.model.output_shape}")

test_image = input("\nEnter path to test image: ")

if not os.path.exists(test_image):
    print(f"❌ ERROR: Image not found: {test_image}")
    exit(1)

print("\n🔍 Running analysis...")
print("="*70)

try:
    # Load and preprocess image
    print(f"\n1️⃣  Loading image: {test_image}")
    img = cv2.imread(test_image)
    if img is None:
        raise ValueError(f"Cannot read image: {test_image}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (detector.image_size, detector.image_size))
    img = img.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img, axis=0)
    
    print(f"   ✅ Image loaded and preprocessed")
    print(f"   Input shape: {img_batch.shape}")
    
    # Make raw prediction
    print(f"\n2️⃣  Making raw prediction...")
    raw_prediction = detector.model.predict(img_batch, verbose=0)
    
    print(f"   Raw prediction output: {raw_prediction}")
    print(f"   Output shape: {raw_prediction.shape}")
    
    # Extract probabilities based on model type
    if detector.is_binary_model:
        print(f"\n   ℹ️  Binary model detected (1 output)")
        # Binary: single value 0-1 (0=real, 1=fake)
        fake_prob = float(raw_prediction[0][0]) * 100
        real_prob = (1 - raw_prediction[0][0]) * 100
        
        print(f"\n3️⃣  Probability breakdown:")
        print(f"   Real probability: {real_prob:.2f}%")
        print(f"   Fake probability: {fake_prob:.2f}%")
    else:
        print(f"\n   ℹ️  Categorical model detected (2 outputs)")
        # Categorical: [real_prob, fake_prob]
        real_prob = float(raw_prediction[0][0]) * 100
        fake_prob = float(raw_prediction[0][1]) * 100
        
        print(f"\n3️⃣  Probability breakdown:")
        print(f"   Index [0] (REAL): {real_prob:.2f}%")
        print(f"   Index [1] (FAKE): {fake_prob:.2f}%")
    
    # Determine prediction
    is_deepfake = fake_prob > real_prob
    confidence = max(real_prob, fake_prob)
    
    print(f"\n4️⃣  Raw determination:")
    print(f"   Is Deepfake: {is_deepfake}")
    print(f"   Confidence: {confidence:.2f}%")
    
    # Full prediction using detector
    print(f"\n5️⃣  Running full detector prediction...")
    import time
    start_time = time.time()
    result = detector.predict(test_image, generate_heatmap=True)
    process_time = time.time() - start_time
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print("="*70)
    
    print(f"\n📊 PREDICTION RESULTS:")
    print(f"   Verdict: {'🚨 DEEPFAKE DETECTED' if result['is_fake'] else '✅ AUTHENTIC IMAGE'}")
    print(f"   Confidence: {result['confidence']*100:.2f}%")
    print(f"   Real Probability: {result['real_probability']*100:.2f}%")
    print(f"   Fake Probability: {result['fake_probability']*100:.2f}%")
    print(f"   Process Time: {process_time:.3f}s")
    print(f"   Prediction: {result['prediction']}")
    
    print(f"\n🔥 Heatmap Generation:")
    if result.get('heatmap'):
        print(f"   ✅ Generated (PIL Image object)")
        print(f"   Size: {result['heatmap'].size}")
        
        # Save heatmap for viewing
        heatmap_save_path = test_image.replace('.', '_heatmap.')
        result['heatmap'].save(heatmap_save_path)
        print(f"   💾 Saved to: {heatmap_save_path}")
    else:
        print(f"   ❌ Not generated")
    
    # Compare raw vs processed
    print(f"\n🔍 VERIFICATION:")
    raw_is_fake = fake_prob > real_prob
    processed_is_fake = result['is_fake']
    
    if raw_is_fake == processed_is_fake:
        print(f"   ✅ Raw and processed predictions MATCH")
    else:
        print(f"   ⚠️  Raw and processed predictions DIFFER")
        print(f"      Raw: {'FAKE' if raw_is_fake else 'REAL'}")
        print(f"      Processed: {'FAKE' if processed_is_fake else 'REAL'}")
    
    # Interpretation guide
    print(f"\n📖 INTERPRETATION GUIDE:")
    if result['is_fake']:
        print(f"   🚨 This image appears to be MANIPULATED/AI-GENERATED")
        if result['confidence'] > 0.9:
            print(f"   💪 Very high confidence - likely a clear deepfake")
        elif result['confidence'] > 0.7:
            print(f"   👍 High confidence - probable deepfake")
        else:
            print(f"   🤔 Moderate confidence - manual review recommended")
    else:
        print(f"   ✅ This image appears to be AUTHENTIC")
        if result['confidence'] > 0.9:
            print(f"   💪 Very high confidence - likely genuine")
        elif result['confidence'] > 0.7:
            print(f"   👍 High confidence - probably authentic")
        else:
            print(f"   🤔 Moderate confidence - manual review recommended")
    
    print(f"\n💡 TROUBLESHOOTING:")
    print(f"   If known real images show as fake (or vice versa):")
    print(f"   1. Check training data labels are correct")
    print(f"   2. Verify class order in training: ['real', 'fake']")
    print(f"   3. Model might need more training data")
    print(f"   4. Current accuracy with 400 images: 70-85% (expect some errors)")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")
    
except Exception as e:
    print(f"\n❌ TEST FAILED!")
    print(f"   Error: {e}")
    print("\n" + "="*70)
    import traceback
    traceback.print_exc()