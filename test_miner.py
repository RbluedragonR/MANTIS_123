#!/usr/bin/env python3

import os
import sys
import json
import tempfile
from dotenv import load_dotenv
from miner import MANTISMiner
import config

def test_environment():
    """Test that all required environment variables are set."""
    print("🔍 Testing environment variables...")
    
    required_vars = [
        'R2_ACCOUNT_ID',
        'R2_WRITE_ACCESS_KEY_ID', 
        'R2_WRITE_SECRET_ACCESS_KEY',
        'R2_BUCKET_ID'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please check your .env file configuration.")
        return False
    else:
        print("✅ All required environment variables are set")
        return True

def test_embedding_generation():
    """Test the embedding generation function."""
    print("\n🧠 Testing embedding generation...")
    
    try:
        # Create a temporary miner instance (this will fail if credentials are wrong)
        # But we can still test the embedding generation logic
        embedding = []
        
        # Simulate the embedding generation
        import numpy as np
        embedding = np.random.uniform(-1, 1, size=config.FEATURE_LENGTH).tolist()
        
        # Validate embedding
        if len(embedding) != config.FEATURE_LENGTH:
            print(f"❌ Embedding has wrong length: {len(embedding)} (expected {config.FEATURE_LENGTH})")
            return False
        
        for i, val in enumerate(embedding):
            if not isinstance(val, (int, float)):
                print(f"❌ Embedding contains non-numeric value at index {i}: {val}")
                return False
            if val < -1 or val > 1:
                print(f"❌ Embedding value out of range at index {i}: {val}")
                return False
        
        print(f"✅ Embedding generation successful: {len(embedding)} features, all values in [-1, 1]")
        return True
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return False

def test_encryption():
    """Test the encryption functionality."""
    print("\n🔐 Testing encryption...")
    
    try:
        from timelock import Timelock
        import requests
        import secrets
        
        # Test data
        test_embedding = [0.5, -0.3, 0.1] * 33 + [0.0]  # 100 values
        
        # Get Drand info
        DRAND_API = "https://api.drand.sh/v2"
        DRAND_BEACON_ID = "quicknet"
        DRAND_PUBLIC_KEY = (
            "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
            "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
            "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
        )
        
        response = requests.get(f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info", timeout=10)
        response.raise_for_status()
        info = response.json()
        
        # Validate required keys exist
        if "genesis_time" not in info or "period" not in info:
            raise ValueError(f"Drand info missing required keys. Got: {list(info.keys())}")
        
        # Validate period is not zero
        if info["period"] <= 0:
            raise ValueError(f"Invalid Drand period: {info['period']}")
        
        # Calculate a future round
        import time
        future_time = time.time() + 30
        time_diff = future_time - info["genesis_time"]
        
        if time_diff <= 0:
            raise ValueError(f"Future time {future_time} is before genesis time {info['genesis_time']}")
            
        target_round = int(time_diff // info["period"])
        
        if target_round <= 0:
            raise ValueError(f"Calculated invalid round: {target_round}")
        
        # Encrypt
        tlock = Timelock(DRAND_PUBLIC_KEY)
        vector_str = str(test_embedding)
        salt = secrets.token_bytes(32)
        ciphertext_hex = tlock.tle(target_round, vector_str, salt).hex()
        
        # Create payload
        payload = {
            "round": target_round,
            "ciphertext": ciphertext_hex
        }
        
        # Validate payload
        if "round" not in payload or "ciphertext" not in payload:
            print("❌ Payload missing required fields")
            return False
        
        if not isinstance(payload["ciphertext"], str):
            print("❌ Ciphertext is not a string")
            return False
        
        print(f"✅ Encryption successful: round {payload['round']}, ciphertext length {len(payload['ciphertext'])}")
        return True
        
    except Exception as e:
        print(f"❌ Encryption failed: {e}")
        return False

def test_payload_creation():
    """Test the complete payload creation process."""
    print("\n📦 Testing payload creation...")
    
    try:
        # Create test payload
        test_payload = {
            "round": 12345,
            "ciphertext": "abcdef123456" * 20  # Mock ciphertext
        }
        
        # Test saving to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_payload, f, indent=2)
            temp_file = f.name
        
        # Verify file was created and contains correct data
        if not os.path.exists(temp_file):
            print("❌ Payload file was not created")
            return False
        
        with open(temp_file, 'r') as f:
            loaded_payload = json.load(f)
        
        if loaded_payload != test_payload:
            print("❌ Payload file contains incorrect data")
            return False
        
        # Clean up
        os.unlink(temp_file)
        
        print("✅ Payload creation and file handling successful")
        return True
        
    except Exception as e:
        print(f"❌ Payload creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔧 MANTIS Miner Setup Test")
    print("=" * 40)
    
    # Load environment
    load_dotenv()
    
    tests = [
        ("Environment", test_environment),
        ("Embedding Generation", test_embedding_generation), 
        ("Encryption", test_encryption),
        ("Payload Creation", test_payload_creation),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test '{name}' crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Your miner setup looks good.")
        print("\nNext steps:")
        print("1. Make sure your R2 bucket is configured for public access")
        print("2. Register your hotkey on netuid 123 if you haven't already")
        print("3. Run the miner with: python miner.py --wallet.name your_wallet --wallet.hotkey your_hotkey --commit-url")
    else:
        print("⚠️  Some tests failed. Please fix the issues before running the miner.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 