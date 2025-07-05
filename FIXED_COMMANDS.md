# MANTIS Miner - Fixed Commands Reference

## 🔧 **Critical Issues Fixed**

Your developer identified and we've fixed these important issues:

1. ✅ **Argument names with dots** - Changed to underscores
2. ✅ **Incorrect getattr usage** - Now using direct attribute access
3. ✅ **Drand API error handling** - Added validation and error checks
4. ✅ **File race conditions** - Using unique temporary files

## 🚀 **Updated Commands**

### **Training Commands**

#### **Train Model Only** (Recommended first step)
```bash
python miner.py --wallet_name your_wallet --wallet_hotkey your_hotkey --train_only
```

#### **Force Retrain** (with fresh data)
```bash
python miner.py --wallet_name your_wallet --wallet_hotkey your_hotkey --retrain
```

### **Mining Commands**

#### **Start Mining** (with URL commit)
```bash
python miner.py --wallet_name your_wallet --wallet_hotkey your_hotkey --commit_url
```

#### **Regular Mining** (after setup)
```bash
python miner.py --wallet_name your_wallet --wallet_hotkey your_hotkey --interval 60
```

#### **Custom R2 URL** (if auto-detection fails)
```bash
python miner.py --wallet_name your_wallet --wallet_hotkey your_hotkey --custom_url "https://your-bucket.r2.dev/your_hotkey"
```

## 📋 **Fixed Command Line Arguments**

| **Old (Broken)** | **New (Fixed)** | **Description** |
|-------------------|-----------------|-----------------|
| `--wallet.name` | `--wallet_name` | Wallet name (required) |
| `--wallet.hotkey` | `--wallet_hotkey` | Hotkey name (required) |
| `--commit-url` | `--commit_url` | Commit URL to subnet |
| `--custom-url` | `--custom_url` | Custom R2 public URL |
| `--train-only` | `--train_only` | Only train, don't mine |

## 🛡️ **Improved Error Handling**

### **Drand API Validation**
Now includes proper validation for:
- HTTP response status codes
- Required JSON keys (`genesis_time`, `period`)
- Division by zero prevention
- Invalid time calculations

### **File Safety**
- **Unique temporary files** prevent race conditions
- **Process ID + UUID** in filenames
- **Automatic cleanup** on success and error
- **No file conflicts** between parallel miners

## 🧪 **Testing Commands**

### **Test Setup**
```bash
python test_miner.py
```

### **Test Prediction Model**
```bash
python test_prediction.py
```

## 📊 **Example Usage Workflow**

```bash
# 1. Setup environment
cd /home/shadeform/MANTIS_123
./install_reqs.sh
source .venv/bin/activate
pip install yfinance ccxt pandas-ta

# 2. Test everything works
python test_prediction.py

# 3. Train your model first
python miner.py --wallet_name my_wallet --wallet_hotkey my_hotkey --train_only

# 4. Start mining (first time with URL commit)
python miner.py --wallet_name my_wallet --wallet_hotkey my_hotkey --commit_url --interval 60

# 5. Regular mining (subsequent runs)
python miner.py --wallet_name my_wallet --wallet_hotkey my_hotkey --interval 60
```

## 🐛 **Common Issues & Solutions**

### **Drand API Errors**
```bash
# If you see "Drand info missing required keys"
# This is now properly handled with fallback behavior
```

### **File Permission Errors**
```bash
# If you get permission denied errors
chmod 755 /tmp  # Ensure temp directory is writable

# Check temp directory
echo $TMPDIR
ls -la /tmp/
```

### **Race Condition Issues**
```bash
# Multiple miners with same hotkey
# Now uses unique temp files: hotkey_PID_UUID.tmp
# Example: 5DHo...WYs_12345_ab3d47f2.tmp
```

## ⚠️ **Breaking Changes**

If you were using the old command format, you MUST update:

```bash
# OLD (Will fail)
python miner.py --wallet.name test --wallet.hotkey test --commit-url

# NEW (Works correctly)
python miner.py --wallet_name test --wallet_hotkey test --commit_url
```

## 🔍 **Debug Mode**

For detailed logging and troubleshooting:

```bash
python miner.py --wallet_name your_wallet --wallet_hotkey your_hotkey --interval 60 2>&1 | tee miner.log
```

## 💡 **Best Practices**

1. **Always train first** with `--train_only`
2. **Use unique process names** if running multiple miners
3. **Monitor temp directory usage** (`df -h /tmp`)
4. **Check logs regularly** for error patterns
5. **Test with prediction test** before mining

---

## 🎯 **All Fixed Issues Summary**

✅ **Fixed argparse dot notation** → underscore arguments
✅ **Fixed getattr usage** → direct attribute access  
✅ **Added Drand validation** → prevents API errors
✅ **Added file safety** → prevents race conditions
✅ **Improved error handling** → better debugging
✅ **Updated documentation** → correct commands

**Your miner is now robust and production-ready!** 🚀 