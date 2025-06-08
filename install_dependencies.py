#!/usr/bin/env python3
"""
DataWeaver-AI Dependency Installation Script

This script helps install all required packages and validates the installation.
Run this after creating your virtual environment to ensure all dependencies
are properly installed and compatible.
"""

import subprocess
import sys
import os
from pathlib import Path

def print_status(message):
    """Print status message with formatting"""
    print(f"\n🔧 {message}")

def print_success(message):
    """Print success message with formatting"""
    print(f"✅ {message}")

def print_warning(message):
    """Print warning message with formatting"""
    print(f"⚠️  {message}")

def print_error(message):
    """Print error message with formatting"""
    print(f"❌ {message}")

def check_python_version():
    """Check if Python version is 3.9 or higher"""
    print_status("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print_error(f"Python {version.major}.{version.minor} detected. Python 3.9+ required!")
        print("Please upgrade Python to 3.9 or higher and try again.")
        return False
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro} - Compatible ✓")
    return True

def upgrade_pip():
    """Upgrade pip to latest version"""
    print_status("Upgrading pip...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        print_success("pip upgraded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to upgrade pip: {e}")
        return False

def install_requirements():
    """Install packages from requirements.txt"""
    print_status("Installing packages from requirements.txt...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print_error("requirements.txt not found!")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            '-r', 'requirements.txt',
            '--upgrade'
        ])
        print_success("All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install packages: {e}")
        print_warning("Trying with --no-cache-dir...")
        
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                '-r', 'requirements.txt',
                '--no-cache-dir', '--upgrade'
            ])
            print_success("Packages installed successfully (no cache)")
            return True
        except subprocess.CalledProcessError as e2:
            print_error(f"Installation failed even with --no-cache-dir: {e2}")
            return False

def test_critical_imports():
    """Test importing critical packages"""
    print_status("Testing critical package imports...")
    
    critical_packages = [
        ('streamlit', 'Streamlit web framework'),
        ('pandas', 'Data manipulation'),
        ('numpy', 'Numerical computing'),
        ('matplotlib', 'Basic plotting'),
        ('plotly', 'Interactive plotting'),
        ('openai', 'OpenAI API'),
        ('langchain', 'LangChain framework'),
        ('langchain_community', 'LangChain community'),
        ('langchain_openai', 'LangChain OpenAI'),
    ]
    
    success_count = 0
    for package, description in critical_packages:
        try:
            __import__(package)
            print_success(f"{package} - {description}")
            success_count += 1
        except ImportError as e:
            print_error(f"{package} - {description}: {e}")
    
    print(f"\n📊 Critical packages: {success_count}/{len(critical_packages)} working")
    return success_count == len(critical_packages)

def test_optional_imports():
    """Test importing optional enhancement packages"""
    print_status("Testing optional enhancement packages...")
    
    optional_packages = [
        ('wordcloud', 'Word cloud generation'),
        ('selenium', 'Web automation'),
        ('beautifulsoup4', 'Web scraping'),
        ('pytesseract', 'OCR processing'),
        ('pdf2image', 'PDF to image conversion'),
        ('statsmodels', 'Statistical analysis'),
        ('sklearn', 'Machine learning'),
        ('duckduckgo_search', 'Web search'),
        ('rich', 'Enhanced terminal output'),
        ('loguru', 'Enhanced logging'),
    ]
    
    success_count = 0
    for package, description in optional_packages:
        try:
            if package == 'beautifulsoup4':
                __import__('bs4')  # BeautifulSoup imports as bs4
            elif package == 'duckduckgo_search':
                __import__('duckduckgo_search')
            else:
                __import__(package)
            print_success(f"{package} - {description}")
            success_count += 1
        except ImportError:
            print_warning(f"{package} - {description} (optional)")
    
    print(f"\n📊 Optional packages: {success_count}/{len(optional_packages)} working")
    return success_count

def check_system_dependencies():
    """Check for system dependencies"""
    print_status("Checking system dependencies...")
    
    print("\n🔍 OCR Dependencies (for PDF processing):")
    try:
        subprocess.check_output(['tesseract', '--version'], stderr=subprocess.STDOUT)
        print_success("Tesseract OCR found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_warning("Tesseract OCR not found - install with:")
        print("  macOS: brew install tesseract")
        print("  Ubuntu: sudo apt-get install tesseract-ocr")
        print("  Windows: Download from https://github.com/tesseract-ocr/tesseract")
    
    print("\n📄 PDF Processing Dependencies:")
    try:
        subprocess.check_output(['pdftoppm', '-h'], stderr=subprocess.STDOUT)
        print_success("Poppler (pdftoppm) found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_warning("Poppler not found - install with:")
        print("  macOS: brew install poppler")
        print("  Ubuntu: sudo apt-get install poppler-utils")
        print("  Windows: Download from https://poppler.freedesktop.org/")

def main():
    """Main installation function"""
    print("🚀 DataWeaver-AI Dependency Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Upgrade pip
    if not upgrade_pip():
        print_warning("pip upgrade failed, continuing anyway...")
    
    # Install requirements
    if not install_requirements():
        print_error("Failed to install required packages!")
        sys.exit(1)
    
    # Test imports
    critical_success = test_critical_imports()
    optional_count = test_optional_imports()
    
    # Check system dependencies
    check_system_dependencies()
    
    # Final status
    print("\n" + "=" * 50)
    if critical_success:
        print_success("✅ Installation completed successfully!")
        print_success("All critical packages are working properly.")
        print(f"📊 {optional_count} optional enhancement packages available.")
        print("\n🚀 You can now run: streamlit run src/app.py")
    else:
        print_error("❌ Installation completed with errors!")
        print("Some critical packages failed to install or import.")
        print("Please check the error messages above and resolve them.")
        sys.exit(1)

if __name__ == "__main__":
    main() 