#!/usr/bin/env python3

import json
import random
import secrets
import time
import os
import argparse
import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import bittensor as bt
from dotenv import load_dotenv
from timelock import Timelock

import comms
import config
from btc_predictor import BTCPredictor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Drand configuration
DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)

class MANTISMiner:
    def __init__(self, wallet_name: str, hotkey_name: str, netuid: int = 123):
        self.wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
        self.subtensor = bt.subtensor(network="finney")
        self.netuid = netuid
        self.hotkey = self.wallet.hotkey.ss58_address
        self.tlock = Timelock(DRAND_PUBLIC_KEY)
        
        # Initialize Bitcoin predictor
        self.btc_predictor = BTCPredictor(model_path=f"btc_model_{self.hotkey[:8]}.pth")
        
        # Ensure required env vars are set
        required_env_vars = ['R2_ACCOUNT_ID', 'R2_WRITE_ACCESS_KEY_ID', 'R2_WRITE_SECRET_ACCESS_KEY']
        for var in required_env_vars:
            if not os.getenv(var):
                raise ValueError(f"Missing required environment variable: {var}")
        
        logger.info(f"Initialized MANTIS miner with hotkey: {self.hotkey}")
        
        # Train model if not already trained
        if self.btc_predictor.model is None:
            logger.info("No trained model found. Training new Bitcoin prediction model...")
            try:
                self.btc_predictor.train_model()
                logger.info("✅ Model training completed successfully")
            except Exception as e:
                logger.error(f"❌ Model training failed: {e}")
                logger.warning("Will use fallback random predictions")
    
    def generate_embedding(self) -> list[float]:
        """
        Generate Bitcoin price prediction embedding using trained model.
        
        Returns:
            A list of 100 floats between -1 and 1 representing prediction signals
        """
        try:
            # Use the Bitcoin predictor to generate embedding
            embedding = self.btc_predictor.generate_mantis_embedding()
            
            logger.info(f"Generated prediction embedding with {len(embedding)} features")
            logger.info(f"Embedding stats: min={min(embedding):.3f}, max={max(embedding):.3f}, mean={np.mean(embedding):.3f}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate prediction embedding: {e}")
            logger.warning("Falling back to random embedding")
            
            # Fallback to random embedding if prediction fails
            embedding = np.random.uniform(-1, 1, size=config.FEATURE_LENGTH).tolist()
            logger.info(f"Generated fallback embedding with {len(embedding)} features")
            return embedding
    
    def encrypt_payload(self, embedding: list[float], lock_time_seconds: int = 30) -> dict:
        """
        Encrypt the embedding for future decryption using tlock.
        
        Args:
            embedding: The 100-dimensional embedding to encrypt
            lock_time_seconds: How many seconds in the future to lock for
        
        Returns:
            Dictionary containing round number and ciphertext
        """
        try:
            # Get Drand beacon info with proper error handling
            response = requests.get(f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info", timeout=10)
            response.raise_for_status()
            info = response.json()
            
            # Validate required keys exist
            if "genesis_time" not in info or "period" not in info:
                raise ValueError(f"Drand info missing required keys. Got: {list(info.keys())}")
            
            # Validate period is not zero
            if info["period"] <= 0:
                raise ValueError(f"Invalid Drand period: {info['period']}")
            
            # Calculate target round for future decryption
            future_time = time.time() + lock_time_seconds
            time_diff = future_time - info["genesis_time"]
            
            if time_diff <= 0:
                raise ValueError(f"Future time {future_time} is before genesis time {info['genesis_time']}")
                
            target_round = int(time_diff // info["period"])
            
            if target_round <= 0:
                raise ValueError(f"Calculated invalid round: {target_round}")
            
            # Encrypt the embedding
            vector_str = str(embedding)
            salt = secrets.token_bytes(32)
            ciphertext_hex = self.tlock.tle(target_round, vector_str, salt).hex()
            
            payload = {
                "round": target_round,
                "ciphertext": ciphertext_hex
            }
            
            logger.info(f"Encrypted payload for round {target_round}")
            return payload
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch Drand info: {e}")
            raise
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid Drand response: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to encrypt payload: {e}")
            raise
    
    def save_and_upload_payload(self, payload: dict) -> bool:
        """
        Save payload to file and upload to R2.
        
        Args:
            payload: The encrypted payload dictionary
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create unique temporary file to prevent race conditions
            temp_dir = tempfile.gettempdir()
            unique_id = str(uuid.uuid4())[:8]
            process_id = os.getpid()
            temp_filename = f"{self.hotkey}_{process_id}_{unique_id}.tmp"
            temp_filepath = os.path.join(temp_dir, temp_filename)
            
            # Save to temporary file first
            with open(temp_filepath, 'w') as f:
                json.dump(payload, f, indent=2)
            
            logger.info(f"Saved payload to temporary file: {temp_filename}")
            
            # Upload to R2 bucket
            bucket_name = comms.bucket()
            if not bucket_name:
                logger.error("R2_BUCKET_ID not configured")
                # Clean up temp file
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                return False
            
            # Use the comms module to upload (R2 object key MUST be hotkey)
            comms.upload(bucket_name, self.hotkey, temp_filepath)
            logger.info(f"Uploaded payload to R2 bucket: {bucket_name}")
            
            # Clean up temporary file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
                logger.debug(f"Cleaned up temporary file: {temp_filepath}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save/upload payload: {e}")
            # Clean up temp file on error
            try:
                temp_filepath = locals().get('temp_filepath')
                if temp_filepath and os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                    logger.debug(f"Cleaned up temporary file after error: {temp_filepath}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temp file: {cleanup_error}")
            return False
    
    def commit_url_to_subnet(self, custom_url: str = None) -> bool:
        """
        Commit your R2 public URL to the subnet.
        This only needs to be done once (or when your URL changes).
        
        Args:
            custom_url: Optional custom public URL. If not provided, will attempt to construct automatically.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if custom_url:
                r2_public_url = custom_url
            else:
                # Construct your public R2 URL
                # NOTE: You may need to modify this based on your R2 setup
                # Common formats:
                # - https://pub-{hash}.r2.dev/{hotkey}
                # - https://{bucket}.r2.dev/{hotkey}
                # - https://{custom-domain}/{hotkey}
                
                bucket_name = comms.bucket()
                if not bucket_name:
                    logger.error("R2_BUCKET_ID not configured. Cannot construct URL.")
                    return False
                
                # Try common R2 URL formats
                # You should replace this with your actual public URL format
                r2_public_url = f"https://{bucket_name}.r2.dev/{self.hotkey}"
                
                logger.warning(f"Using auto-constructed URL: {r2_public_url}")
                logger.warning("If this is incorrect, use --custom-url flag or modify the URL format in the code")
            
            # Commit to subnet
            success = self.subtensor.commit(
                wallet=self.wallet,
                netuid=self.netuid,
                data=r2_public_url
            )
            
            if success:
                logger.info(f"Successfully committed URL to subnet: {r2_public_url}")
                return True
            else:
                logger.error("Failed to commit URL to subnet")
                return False
                
        except Exception as e:
            logger.error(f"Failed to commit URL: {e}")
            return False
    
    def mine_once(self) -> bool:
        """
        Execute one mining cycle: generate -> encrypt -> upload.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding
            embedding = self.generate_embedding()
            
            # Encrypt payload
            payload = self.encrypt_payload(embedding)
            
            # Save and upload
            success = self.save_and_upload_payload(payload)
            
            if success:
                logger.info("Mining cycle completed successfully")
            else:
                logger.error("Mining cycle failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Mining cycle failed: {e}")
            return False
    
    def retrain_model(self):
        """Retrain the Bitcoin prediction model with fresh data."""
        logger.info("Retraining Bitcoin prediction model...")
        try:
            self.btc_predictor.train_model(retrain=True)
            logger.info("✅ Model retraining completed successfully")
        except Exception as e:
            logger.error(f"❌ Model retraining failed: {e}")
    
    def run(self, interval_seconds: int = 60, commit_url: bool = False, custom_url: str = None):
        """
        Run the mining loop continuously.
        
        Args:
            interval_seconds: How often to mine (in seconds)
            commit_url: Whether to commit URL to subnet on startup
            custom_url: Optional custom public URL for committing
        """
        logger.info(f"Starting MANTIS miner with {interval_seconds}s interval")
        
        if commit_url:
            logger.info("Committing URL to subnet...")
            if not self.commit_url_to_subnet(custom_url):
                logger.error("Failed to commit URL - continuing anyway")
        
        try:
            while True:
                start_time = time.time()
                
                # Execute mining cycle
                success = self.mine_once()
                
                if success:
                    logger.info("✅ Mining cycle successful")
                else:
                    logger.error("❌ Mining cycle failed")
                
                # Wait for next interval
                elapsed = time.time() - start_time
                sleep_time = max(0, interval_seconds - elapsed)
                
                if sleep_time > 0:
                    logger.info(f"Sleeping for {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error in mining loop: {e}")
        finally:
            logger.info("Miner stopped")


def main():
    parser = argparse.ArgumentParser(description="MANTIS Bittensor Miner")
    parser.add_argument("--wallet_name", required=True, help="Wallet name")
    parser.add_argument("--wallet_hotkey", required=True, help="Wallet hotkey")
    parser.add_argument("--netuid", type=int, default=123, help="Subnet netuid")
    parser.add_argument("--interval", type=int, default=60, help="Mining interval in seconds")
    parser.add_argument("--commit_url", action="store_true", help="Commit URL to subnet on startup")
    parser.add_argument("--custom_url", help="Custom public R2 URL (if auto-detection fails)")
    parser.add_argument("--retrain", action="store_true", help="Force retrain the prediction model")
    parser.add_argument("--train_only", action="store_true", help="Only train the model, don't start mining")
    
    args = parser.parse_args()
    
    try:
        miner = MANTISMiner(
            wallet_name=args.wallet_name,
            hotkey_name=args.wallet_hotkey,
            netuid=args.netuid
        )
        
        # Handle training options
        if args.retrain:
            miner.retrain_model()
        
        if args.train_only:
            logger.info("Training complete. Exiting (--train-only specified).")
            return 0
        
        miner.run(
            interval_seconds=args.interval,
            commit_url=args.commit_url,
            custom_url=args.custom_url
        )
        
    except Exception as e:
        logger.error(f"Failed to start miner: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 