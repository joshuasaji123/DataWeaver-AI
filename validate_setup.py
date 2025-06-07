#!/usr/bin/env python3
"""
Quick validation script for Multi-Agent Data Analysis System Enhanced Features
Run this to check if your system is ready for the enhanced setup process.
"""

import sys
import subprocess
import platform
import shutil

def print_status(message, status="INFO"):
    """Print colored status messages"""
    colors = {
        "INFO": "\033[0;34m",
        "SUCCESS": "\033[0;32m", 
        "WARNING": "\033[1;33m",
        "ERROR": "\033[0;31m",
        "FEATURE": "\033[0;35m"
    }
    reset = "\033[0m"
    print(f"{colors.get(status, '')}[{status}]{reset} {message}")

def check_python_version():
    """Check Python version compatibility"""
    print_status("Checking Python version...", "INFO")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version >= (3, 8):
        print_status(f"Python {version_str} - Compatible ✅", "SUCCESS")
        return True
    else:
        print_status(f"Python {version_str} - Incompatible ❌ (Requires 3.8+)", "ERROR")
        return False

def check_system_requirements():
    """Check system requirements for enhanced features"""
    print_status("Checking system requirements...", "INFO")
    
    system = platform.system()
    print_status(f"Operating System: {system}", "INFO")
    
    # Check for Chrome/Chromium (required for Selenium)
    chrome_browsers = ["google-chrome", "chromium-browser", "chromium", "chrome"]
    chrome_found = False
    
    for browser in chrome_browsers:
        if shutil.which(browser):
            print_status(f"Found {browser} - Web Intelligence features will work ✅", "SUCCESS")
            chrome_found = True
            break
    
    if not chrome_found:
        print_status("Chrome/Chromium not found - Web Intelligence may be limited ⚠️", "WARNING")
        print_status("Install Chrome or Chromium for full web features", "WARNING")
    
    return chrome_found

def check_pip():
    """Check if pip is available"""
    print_status("Checking pip availability...", "INFO")
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True, check=True)
        pip_version = result.stdout.strip()
        print_status(f"pip available: {pip_version} ✅", "SUCCESS")
        return True
    except subprocess.CalledProcessError:
        print_status("pip not available ❌", "ERROR")
        return False

def check_venv_support():
    """Check if venv module is available"""
    print_status("Checking virtual environment support...", "INFO")
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "--help"], 
                      capture_output=True, check=True)
        print_status("Virtual environment support available ✅", "SUCCESS")
        return True
    except subprocess.CalledProcessError:
        print_status("Virtual environment support not available ❌", "ERROR")
        return False

def check_internet_connection():
    """Basic check for internet connectivity"""
    print_status("Checking internet connectivity...", "INFO")
    
    try:
        import urllib.request
        urllib.request.urlopen('https://pypi.org', timeout=10)
        print_status("Internet connection available ✅", "SUCCESS")
        return True
    except:
        print_status("Internet connection issues - may affect installation ⚠️", "WARNING")
        return False

def main():
    """Main validation function"""
    print("🔍 Multi-Agent System - Enhanced Setup Validation")
    print("=" * 55)
    print()
    
    checks = []
    
    # Core requirements
    print_status("Core Requirements:", "FEATURE")
    checks.append(("Python Version", check_python_version()))
    checks.append(("pip Package Manager", check_pip()))
    checks.append(("Virtual Environment", check_venv_support()))
    
    print()
    
    # System requirements
    print_status("System Requirements:", "FEATURE")
    checks.append(("Web Browser Support", check_system_requirements()))
    checks.append(("Internet Connectivity", check_internet_connection()))
    
    print()
    print("=" * 55)
    
    # Summary
    passed = sum(1 for _, status in checks if status)
    total = len(checks)
    
    print_status(f"Validation Summary: {passed}/{total} checks passed", "INFO")
    
    if passed == total:
        print_status("✅ System is ready for enhanced setup!", "SUCCESS")
        print_status("Run ./setup.sh to install all enhanced features", "INFO")
        return 0
    elif passed >= 3:  # Core requirements met
        print_status("⚠️  System meets core requirements but some features may be limited", "WARNING")
        print_status("You can proceed with setup.sh - some enhanced features may not work", "INFO")
        return 0
    else:
        print_status("❌ System does not meet minimum requirements", "ERROR")
        print_status("Please install Python 3.8+, pip, and venv support before proceeding", "ERROR")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 