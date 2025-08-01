#!/usr/bin/env python3
"""
Model Compatibility Fixer
=========================

This script helps fix TensorFlow model compatibility issues by:
1. Loading an existing model with compile=False
2. Recompiling with current TensorFlow version
3. Saving in both Keras and H5 formats

Usage:
    python fix_model_compatibility.py

Requirements:
    - tensorflow>=2.16.0
    - Existing model.h5 file
"""

import os
import tensorflow as tf

def fix_model_compatibility():
    """Fix model compatibility by recompiling and saving in multiple formats."""
    
    print("🔧 TensorFlow Model Compatibility Fixer")
    print("=" * 50)
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check if model file exists
    if not os.path.exists('model.h5'):
        print("❌ Error: model.h5 not found!")
        print("Please run Experiments.ipynb first to train the model.")
        return False
    
    try:
        print("\n📥 Loading existing model...")
        # Load model without compiling to avoid compatibility issues
        model = tf.keras.models.load_model('model.h5', compile=False)
        print("✅ Model loaded successfully")
        
        print("\n🔄 Recompiling with current TensorFlow version...")
        # Recompile with current TensorFlow version
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )
        print("✅ Model recompiled successfully")
        
        print("\n💾 Saving in multiple formats...")
        
        # Save in Keras format (recommended)
        model.save('model.keras')
        print("✅ Saved: model.keras (recommended format)")
        
        # Save in H5 format (backward compatibility)
        model.save('model.h5', save_format='h5')
        print("✅ Saved: model.h5 (legacy format)")
        
        print("\n🎉 Model compatibility fix completed!")
        print("Your Streamlit app should now work properly.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Try running Experiments.ipynb to retrain the model.")
        return False

if __name__ == "__main__":
    success = fix_model_compatibility()
    
    if success:
        print("\n📋 Next Steps:")
        print("1. Restart your Streamlit app")
        print("2. The app should now load the model successfully")
        print("3. If issues persist, retrain using Experiments.ipynb")
    else:
        print("\n📋 Alternative Solutions:")
        print("1. Run all cells in Experiments.ipynb to retrain")
        print("2. Check TensorFlow version compatibility")
        print("3. Update requirements.txt if needed")
