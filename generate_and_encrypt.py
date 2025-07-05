import json
import random
import secrets
import time

import requests
from timelock import Timelock

# --- Configuration ---
FILENAME = "5Dw5G2ECqFDfZFC48yKAiuuoZB81xKdYQ1ZuNVKw4ThgYywZ"
LOCK_TIME_SECONDS = 30
FEATURE_LENGTH = 100

# --- tlock/Drand Configuration ---
DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)

def generate_and_encrypt():
    """
    Generates a random vector, encrypts it for 30 seconds in the future,
    and saves it to a file named after the specified hotkey.
    """
    print("--- Starting Payload Generation ---")

    # 1. Generate random data
    random_vector = [random.uniform(-1, 1) for _ in range(FEATURE_LENGTH)]
    print(f"Generated a random vector of length {len(random_vector)}.")
    
    # 2. Determine future Drand round
    print(f"Fetching Drand beacon info to target a round ~{LOCK_TIME_SECONDS}s in the future...")
    try:
        response = requests.get(f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info", timeout=10)
        response.raise_for_status()
        info = response.json()
        
        # Validate required keys exist
        if "genesis_time" not in info or "period" not in info:
            raise ValueError(f"Drand info missing required keys. Got: {list(info.keys())}")
        
        # Validate period is not zero
        if info["period"] <= 0:
            raise ValueError(f"Invalid Drand period: {info['period']}")
        
        future_time = time.time() + LOCK_TIME_SECONDS
        time_diff = future_time - info["genesis_time"]
        
        if time_diff <= 0:
            raise ValueError(f"Future time {future_time} is before genesis time {info['genesis_time']}")
            
        round_num = int(time_diff // info["period"])
        
        if round_num <= 0:
            raise ValueError(f"Calculated invalid round: {round_num}")
            
        print(f"Targeting Drand round: {round_num}")
    except Exception as e:
        print(f"❌ Error: Could not fetch Drand info. {e}")
        return

    # 3. Encrypt the data
    try:
        tlock = Timelock(DRAND_PUBLIC_KEY)
        vector_str = str(random_vector)
        salt = secrets.token_bytes(32)
        
        ciphertext_hex = tlock.tle(round_num, vector_str, salt).hex()
        print("Encryption successful.")
    except Exception as e:
        print(f"❌ Error: Encryption failed. {e}")
        return

    # 4. Package and save the payload
    payload_dict = {"round": round_num, "ciphertext": ciphertext_hex}
    payload_json = json.dumps(payload_dict, indent=2)

    try:
        with open(FILENAME, "w") as f:
            f.write(payload_json)
        print(f"✅ Success! Encrypted payload saved to file: {FILENAME}")
    except Exception as e:
        print(f"❌ Error: Failed to write to file {FILENAME}. {e}")

if __name__ == "__main__":
    generate_and_encrypt() 