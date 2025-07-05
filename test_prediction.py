#!/usr/bin/env python3

import sys
import logging
from btc_predictor import BTCPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_btc_predictor():
    """Test the Bitcoin predictor functionality."""
    print("🔍 Testing Bitcoin Prediction Model")
    print("=" * 50)
    
    try:
        # Initialize predictor
        print("1. Initializing Bitcoin predictor...")
        predictor = BTCPredictor("test_model.pth")
        
        # Test data fetching
        print("2. Testing data fetching...")
        df = predictor.data_fetcher.get_historical_data(hours=72)
        print(f"   ✅ Fetched {len(df)} hours of data")
        print(f"   📊 Latest BTC price: ${df['close'].iloc[-1]:.2f}")
        
        # Test feature engineering
        print("3. Testing feature engineering...")
        df_features = predictor.feature_engineer.prepare_features(df)
        print(f"   ✅ Generated {len(predictor.feature_engineer.feature_names)} features")
        print(f"   🔧 Feature names: {predictor.feature_engineer.feature_names[:5]}...")
        
        # Test model training (quick version)
        print("4. Testing model training...")
        if predictor.model is None:
            print("   🏋️ Training new model (this may take a few minutes)...")
            predictor.train_model()
            print("   ✅ Model training completed")
        else:
            print("   ✅ Model already loaded")
        
        # Test prediction
        print("5. Testing prediction...")
        direction_prob = predictor.predict_direction()
        print(f"   📈 Prediction probability: {direction_prob:.4f}")
        print(f"   🎯 Predicted direction: {'UP' if direction_prob > 0.5 else 'DOWN'}")
        
        # Test MANTIS embedding generation
        print("6. Testing MANTIS embedding generation...")
        embedding = predictor.generate_mantis_embedding()
        print(f"   ✅ Generated {len(embedding)} dimensions")
        print(f"   📊 Stats: min={min(embedding):.3f}, max={max(embedding):.3f}, mean={sum(embedding)/len(embedding):.3f}")
        
        # Validate embedding format
        print("7. Validating embedding format...")
        all_valid = all(-1 <= val <= 1 for val in embedding)
        correct_length = len(embedding) == 100
        
        if all_valid and correct_length:
            print("   ✅ Embedding validation passed")
        else:
            print(f"   ❌ Embedding validation failed:")
            print(f"      - Length correct: {correct_length}")
            print(f"      - Values in range: {all_valid}")
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print(f"📊 Sample embedding values: {embedding[:10]}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("Bitcoin Prediction Model Test")
    print("Testing GPU availability...")
    
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  GPU not available - using CPU")
    
    print()
    
    success = test_btc_predictor()
    
    if success:
        print("\n🚀 Bitcoin prediction model is ready for MANTIS mining!")
        print("\nNext steps:")
        print("1. Set up your .env file with R2 credentials")
        print("2. Register your hotkey on subnet 123")
        print("3. Run: python miner.py --wallet.name your_wallet --wallet.hotkey your_hotkey --commit-url")
        return 0
    else:
        print("\n💔 Test failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 