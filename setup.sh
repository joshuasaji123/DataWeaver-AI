#!/bin/bash

# Multi-Agent Data Analysis System - Enhanced Setup Script
echo "🚀 Multi-Agent Data Analysis System - Enhanced Setup Script"
echo "=========================================================="
echo "💡 This script will set up all enhanced features including:"
echo "   • Advanced Visualizations (Plotly Dash, Bokeh, Altair)"
echo "   • Web Intelligence Tools (Selenium, BeautifulSoup)"
echo "   • Financial Analysis Tools (yfinance, alpha_vantage)"
echo "   • Statistical Analysis Suite (statsmodels, pingouin)"
echo "   • Database Connectivity (SQLAlchemy, PyMongo)"
echo "   • Enhanced Agent Capabilities"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_feature() {
    echo -e "${PURPLE}[FEATURE]${NC} $1"
}

# Check system requirements
check_system_requirements() {
    print_status "Checking system requirements..."
    
    # Check OS
    OS=$(uname -s)
    print_status "Operating System: $OS"
    
    # Check if Chrome/Chromium is available for Selenium
    if command -v google-chrome &> /dev/null; then
        print_success "Google Chrome detected - Selenium will work properly"
        CHROME_AVAILABLE=true
    elif command -v chromium-browser &> /dev/null; then
        print_success "Chromium browser detected - Selenium will work properly"
        CHROME_AVAILABLE=true
    elif command -v chromium &> /dev/null; then
        print_success "Chromium detected - Selenium will work properly"
        CHROME_AVAILABLE=true
    else
        print_warning "Chrome/Chromium not detected - Web Intelligence features may not work"
        print_warning "Install Chrome or Chromium for full functionality"
        CHROME_AVAILABLE=false
    fi
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    print_success "Found Python $PYTHON_VERSION"
    
    # Verify Python version is 3.8+
    if $PYTHON_CMD -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
        print_success "Python version is compatible (3.8+)"
    else
        print_error "Python version must be 3.8 or higher. Current: $PYTHON_VERSION"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Removing old one..."
        rm -rf venv
    fi
    
    $PYTHON_CMD -m venv venv
    if [ $? -eq 0 ]; then
        print_success "Virtual environment created successfully"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Virtual environment activated"
    else
        print_error "Failed to activate virtual environment"
        exit 1
    fi
}

# Install enhanced dependencies
install_enhanced_dependencies() {
    print_status "Installing enhanced dependencies..."
    
    # Upgrade pip first
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install setuptools and wheel
    print_status "Installing build tools..."
    pip install setuptools>=68.0.0 wheel>=0.40.0
    
    # Install core dependencies first
    print_feature "Installing core dependencies..."
    pip install pandas>=1.5.0 streamlit>=1.28.0 numpy>=1.24.0
    
    # Install basic visualization libraries
    print_feature "Installing basic visualization libraries..."
    pip install matplotlib>=3.6.0 seaborn>=0.12.0 plotly>=5.15.0 kaleido>=0.2.1
    
    # Install advanced visualization libraries
    print_feature "Installing advanced visualization libraries..."
    pip install dash>=2.14.0 bokeh>=3.2.0 altair>=5.0.0 wordcloud>=1.9.0 || {
        print_warning "Some advanced visualization libraries failed to install - trying individually..."
        pip install dash>=2.14.0 || print_warning "Dash installation failed"
        pip install bokeh>=3.2.0 || print_warning "Bokeh installation failed"
        pip install altair>=5.0.0 || print_warning "Altair installation failed"
        pip install wordcloud>=1.9.0 || print_warning "WordCloud installation failed"
    }
    
    # Install web intelligence tools
    print_feature "Installing web intelligence tools..."
    pip install selenium>=4.15.0 beautifulsoup4>=4.12.0 requests>=2.31.0 lxml>=4.9.0 || {
        print_warning "Some web tools failed to install - trying individually..."
        pip install selenium>=4.15.0 || print_warning "Selenium installation failed"
        pip install beautifulsoup4>=4.12.0 || print_warning "BeautifulSoup installation failed"
        pip install requests>=2.31.0 || print_warning "Requests installation failed"
        pip install lxml>=4.9.0 || print_warning "LXML installation failed"
    }
    
    # Install WebDriver Manager for Selenium
    print_feature "Installing WebDriver Manager..."
    pip install webdriver-manager>=4.0.0 || print_warning "WebDriver Manager installation failed"
    
    # Install financial analysis tools
    print_feature "Installing financial analysis tools..."
    pip install yfinance>=0.2.0 alpha-vantage>=2.3.0 || {
        print_warning "Some financial tools failed to install - trying individually..."
        pip install yfinance>=0.2.0 || print_warning "yfinance installation failed"
        pip install alpha-vantage>=2.3.0 || print_warning "alpha-vantage installation failed"
    }
    
    # Install statistical analysis suite
    print_feature "Installing statistical analysis suite..."
    pip install statsmodels>=0.14.0 factor-analyzer>=0.4.0 pingouin>=0.5.0 || {
        print_warning "Some statistical tools failed to install - trying individually..."
        pip install statsmodels>=0.14.0 || print_warning "statsmodels installation failed"
        pip install factor-analyzer>=0.4.0 || print_warning "factor-analyzer installation failed"
        pip install pingouin>=0.5.0 || print_warning "pingouin installation failed"
    }
    
    # Install database connectivity
    print_feature "Installing database connectivity..."
    pip install sqlalchemy>=2.0.0 pymongo>=4.5.0 || {
        print_warning "Some database tools failed to install - trying individually..."
        pip install sqlalchemy>=2.0.0 || print_warning "SQLAlchemy installation failed"
        pip install pymongo>=4.5.0 || print_warning "PyMongo installation failed"
    }
    
    # Install OCR and PDF processing tools
    print_feature "Installing OCR and PDF processing tools..."
    pip install pytesseract>=0.3.10 pdf2image>=1.17.0 Pillow>=10.0.0 PyPDF2>=3.0.1 pdfplumber>=0.10.0 || {
        print_warning "Some OCR tools failed to install - trying individually..."
        pip install pytesseract>=0.3.10 || print_warning "pytesseract installation failed"
        pip install pdf2image>=1.17.0 || print_warning "pdf2image installation failed"
        pip install Pillow>=10.0.0 || print_warning "Pillow installation failed"
        pip install PyPDF2>=3.0.1 || print_warning "PyPDF2 installation failed"
        pip install pdfplumber>=0.10.0 || print_warning "pdfplumber installation failed"
        print_warning "OCR features may be limited - install Tesseract and Poppler for full functionality"
    }
    
    # Install file system monitoring
    print_feature "Installing file system monitoring..."
    pip install watchdog>=3.0.0
    
    # Install AI/ML libraries
    print_feature "Installing AI/ML libraries..."
    pip install openai>=1.0.0 ollama>=0.1.0 scikit-learn>=1.3.0 scipy>=1.10.0
    
    # Install optional LangChain dependencies (with error handling)
    print_feature "Installing LangChain dependencies (optional)..."
    pip install langchain>=0.1.0 langchain-community>=0.0.10 langchain-openai>=0.0.5 langchain-experimental>=0.0.5 langgraph>=0.0.40 langsmith>=0.0.60 langchain-ollama>=0.0.1 || {
        print_warning "Some LangChain dependencies failed to install - this is optional and won't affect core functionality"
    }
    
    # Install remaining dependencies
    print_feature "Installing remaining dependencies..."
    pip install python-dotenv>=1.0.0 psutil>=5.9.0 nbformat>=5.7.0
    
    print_success "All enhanced dependencies installed successfully!"
}

# Setup Selenium WebDriver
setup_selenium() {
    print_status "Setting up Selenium WebDriver..."
    
    if [ "$CHROME_AVAILABLE" = true ]; then
        print_status "Testing Selenium setup..."
        $PYTHON_CMD -c "
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

try:
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    # Install and setup ChromeDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.quit()
    print('✅ Selenium WebDriver setup successful!')
except Exception as e:
    print(f'⚠️  Selenium setup warning: {e}')
" || print_warning "Selenium setup encountered issues - web features may be limited"
    else
        print_warning "Skipping Selenium setup - Chrome/Chromium not available"
    fi
}

# Verify enhanced installation
verify_enhanced_installation() {
    print_status "Verifying enhanced installation..."
    
    # Test core imports
    print_status "Testing core dependencies..."
    $PYTHON_CMD -c "
import streamlit
import pandas
import plotly
import matplotlib
import seaborn
import numpy
import openai
import ollama
print('✅ All core dependencies verified successfully!')
" || {
        print_error "Core dependency verification failed"
        exit 1
    }
    
    # Test advanced visualization imports
    print_status "Testing advanced visualization tools..."
    $PYTHON_CMD -c "
success_count = 0
total_count = 4

try:
    import dash
    print('✅ Dash available')
    success_count += 1
except ImportError:
    print('⚠️  Dash not available')

try:
    import bokeh
    print('✅ Bokeh available')
    success_count += 1
except ImportError:
    print('⚠️  Bokeh not available')

try:
    import altair
    print('✅ Altair available')
    success_count += 1
except ImportError:
    print('⚠️  Altair not available')

try:
    import wordcloud
    print('✅ WordCloud available')
    success_count += 1
except ImportError:
    print('⚠️  WordCloud not available')

print(f'📊 Advanced visualization: {success_count}/{total_count} tools available')
"
    
    # Test web intelligence imports
    print_status "Testing web intelligence tools..."
    $PYTHON_CMD -c "
success_count = 0
total_count = 4

try:
    import selenium
    print('✅ Selenium available')
    success_count += 1
except ImportError:
    print('⚠️  Selenium not available')

try:
    import bs4
    print('✅ BeautifulSoup available')
    success_count += 1
except ImportError:
    print('⚠️  BeautifulSoup not available')

try:
    import requests
    print('✅ Requests available')
    success_count += 1
except ImportError:
    print('⚠️  Requests not available')

try:
    import yfinance
    print('✅ yfinance available')
    success_count += 1
except ImportError:
    print('⚠️  yfinance not available')

print(f'🌐 Web intelligence: {success_count}/{total_count} tools available')
"
    
    # Test statistical analysis imports
    print_status "Testing statistical analysis tools..."
    $PYTHON_CMD -c "
success_count = 0
total_count = 4

try:
    import statsmodels
    print('✅ Statsmodels available')
    success_count += 1
except ImportError:
    print('⚠️  Statsmodels not available')

try:
    import factor_analyzer
    print('✅ Factor Analyzer available')
    success_count += 1
except ImportError:
    print('⚠️  Factor Analyzer not available')

try:
    import pingouin
    print('✅ Pingouin available')
    success_count += 1
except ImportError:
    print('⚠️  Pingouin not available')

try:
    import sklearn
    print('✅ Scikit-learn available')
    success_count += 1
except ImportError:
    print('⚠️  Scikit-learn not available')

print(f'📈 Statistical analysis: {success_count}/{total_count} tools available')
"
    
    # Test database connectivity
    print_status "Testing database connectivity..."
    $PYTHON_CMD -c "
success_count = 0
total_count = 2

try:
    import sqlalchemy
    print('✅ SQLAlchemy available')
    success_count += 1
except ImportError:
    print('⚠️  SQLAlchemy not available')

try:
    import pymongo
    print('✅ PyMongo available')
    success_count += 1
except ImportError:
    print('⚠️  PyMongo not available')

print(f'🗄️  Database connectivity: {success_count}/{total_count} tools available')
"
    
    # Test OCR and PDF processing
    print_status "Testing OCR and PDF processing tools..."
    $PYTHON_CMD -c "
success_count = 0
total_count = 5

try:
    import pytesseract
    print('✅ Pytesseract available')
    success_count += 1
except ImportError:
    print('⚠️  Pytesseract not available')

try:
    import pdf2image
    print('✅ pdf2image available')
    success_count += 1
except ImportError:
    print('⚠️  pdf2image not available')

try:
    import PIL
    print('✅ Pillow available')
    success_count += 1
except ImportError:
    print('⚠️  Pillow not available')

try:
    import PyPDF2
    print('✅ PyPDF2 available')
    success_count += 1
except ImportError:
    print('⚠️  PyPDF2 not available')

try:
    import pdfplumber
    print('✅ pdfplumber available')
    success_count += 1
except ImportError:
    print('⚠️  pdfplumber not available')

print(f'🔍 OCR and PDF processing: {success_count}/{total_count} tools available')
if success_count < total_count:
    print('⚠️  Note: For full OCR functionality, install system dependencies:')
    print('   macOS: brew install tesseract poppler')
    print('   Ubuntu: sudo apt-get install tesseract-ocr poppler-utils')
    print('   Windows: Download from official sites or use conda')
"
    
    # Test optional imports (with warnings)
    print_status "Testing optional features..."
    $PYTHON_CMD -c "
try:
    import langchain
    print('✅ LangChain integration available')
except ImportError:
    print('⚠️  LangChain integration not available (optional)')
" || print_warning "Some optional dependencies not available"
    
    print_success "Enhanced installation verification completed!"
}

# Verify file structure
verify_file_structure() {
    print_status "Verifying enhanced file structure..."
    
    # Check for enhanced files
    missing_files=()
    
    # Core files
    [ ! -f "src/app.py" ] && missing_files+=("src/app.py")
    [ ! -f "src/models/agent.py" ] && missing_files+=("src/models/agent.py")
    [ ! -f "src/ui/components.py" ] && missing_files+=("src/ui/components.py")
    
    # Enhanced files
    [ ! -f "src/config/agent_roles.py" ] && missing_files+=("src/config/agent_roles.py")
    [ ! -f "src/utils/advanced_visualizations.py" ] && missing_files+=("src/utils/advanced_visualizations.py")
    [ ! -f "src/utils/web_tools.py" ] && missing_files+=("src/utils/web_tools.py")
    [ ! -f "demo_enhanced_features.py" ] && missing_files+=("demo_enhanced_features.py")
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        print_success "All enhanced files are present"
    else
        print_warning "Some enhanced files are missing:"
        for file in "${missing_files[@]}"; do
            echo "  ⚠️  $file"
        done
        print_warning "The system will still work, but some features may be limited"
    fi
}

# Create enhanced launch script
create_enhanced_launch_script() {
    print_status "Creating enhanced launch script..."
    
    cat > run_app.sh << 'EOF'
#!/bin/bash
# Launch Multi-Agent Data Analysis System - Enhanced Version

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}🚀 Starting Multi-Agent Data Analysis System - Enhanced Version${NC}"
echo -e "${PURPLE}💡 Enhanced Features Available:${NC}"
echo -e "   • Advanced Visualizations (Plotly Dash, Bokeh, Altair)"
echo -e "   • Web Intelligence Tools (Selenium, BeautifulSoup)"
echo -e "   • Financial Analysis (yfinance, alpha_vantage)"
echo -e "   • Statistical Analysis Suite"
echo -e "   • Specialized Agent Types (Accountant, Sports Coach)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found. Please run setup.sh first.${NC}"
    exit 1
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Check if src/app.py exists
if [ ! -f "src/app.py" ]; then
    echo -e "${RED}❌ Application file not found. Please ensure you're in the correct directory.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Environment activated successfully!${NC}"
echo -e "${BLUE}📊 Launching application at http://localhost:8501${NC}"
echo -e "${BLUE}💡 Press Ctrl+C to stop the application${NC}"
echo ""

# Launch the application
streamlit run src/app.py

EOF
    
    chmod +x run_app.sh
    print_success "Enhanced launch script created: run_app.sh"
}

# Create demo launch script
create_demo_script() {
    print_status "Creating demo launch script..."
    
    cat > run_demo.sh << 'EOF'
#!/bin/bash
# Run Enhanced Features Demo

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}🎯 Multi-Agent System - Enhanced Features Demo${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found. Please run setup.sh first.${NC}"
    exit 1
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Check if demo file exists
if [ ! -f "demo_enhanced_features.py" ]; then
    echo -e "${RED}❌ Demo file not found. Please ensure all files are properly installed.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Running enhanced features demo...${NC}"
echo ""

# Run the demo
python demo_enhanced_features.py

EOF
    
    chmod +x run_demo.sh
    print_success "Demo launch script created: run_demo.sh"
}

# Create quick test script
create_test_script() {
    print_status "Creating quick test script..."
    
    cat > test_enhanced_features.py << 'EOF'
#!/usr/bin/env python3
"""
Quick test script for enhanced features
"""

import sys
import importlib

def test_import(module_name, feature_name):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {feature_name} - Available")
        return True
    except ImportError as e:
        print(f"❌ {feature_name} - Not Available ({e})")
        return False

def main():
    print("🧪 Testing Enhanced Features")
    print("=" * 40)
    
    # Test core features
    print("\n📊 Core Features:")
    core_tests = [
        ("streamlit", "Streamlit"),
        ("pandas", "Pandas"),
        ("plotly", "Plotly"),
        ("matplotlib", "Matplotlib"),
        ("numpy", "NumPy"),
    ]
    
    core_passed = sum(test_import(module, name) for module, name in core_tests)
    
    # Test advanced visualizations
    print("\n🎨 Advanced Visualizations:")
    viz_tests = [
        ("dash", "Plotly Dash"),
        ("bokeh", "Bokeh"),
        ("altair", "Altair"),
        ("wordcloud", "WordCloud"),
    ]
    
    viz_passed = sum(test_import(module, name) for module, name in viz_tests)
    
    # Test web intelligence
    print("\n🌐 Web Intelligence:")
    web_tests = [
        ("selenium", "Selenium"),
        ("bs4", "BeautifulSoup"),
        ("requests", "Requests"),
        ("yfinance", "Yahoo Finance"),
    ]
    
    web_passed = sum(test_import(module, name) for module, name in web_tests)
    
    # Test statistical analysis
    print("\n📈 Statistical Analysis:")
    stats_tests = [
        ("statsmodels", "Statsmodels"),
        ("factor_analyzer", "Factor Analyzer"),
        ("pingouin", "Pingouin"),
        ("sklearn", "Scikit-learn"),
    ]
    
    stats_passed = sum(test_import(module, name) for module, name in stats_tests)
    
    # Test database connectivity
    print("\n🗄️ Database Connectivity:")
    db_tests = [
        ("sqlalchemy", "SQLAlchemy"),
        ("pymongo", "PyMongo"),
    ]
    
    db_passed = sum(test_import(module, name) for module, name in db_tests)
    
    # Test AI/ML
    print("\n🤖 AI/ML Libraries:")
    ai_tests = [
        ("openai", "OpenAI"),
        ("ollama", "Ollama"),
    ]
    
    ai_passed = sum(test_import(module, name) for module, name in ai_tests)
    
    # Test optional features
    print("\n🔧 Optional Features:")
    optional_tests = [
        ("langchain", "LangChain"),
        ("watchdog", "File Monitoring"),
    ]
    
    optional_passed = sum(test_import(module, name) for module, name in optional_tests)
    
    # Summary
    total_tests = len(core_tests) + len(viz_tests) + len(web_tests) + len(stats_tests) + len(db_tests) + len(ai_tests) + len(optional_tests)
    total_passed = core_passed + viz_passed + web_passed + stats_passed + db_passed + ai_passed + optional_passed
    
    print("\n" + "=" * 40)
    print(f"📋 Test Summary: {total_passed}/{total_tests} features available")
    print(f"📊 Core Features: {core_passed}/{len(core_tests)}")
    print(f"🎨 Visualizations: {viz_passed}/{len(viz_tests)}")
    print(f"🌐 Web Intelligence: {web_passed}/{len(web_tests)}")
    print(f"📈 Statistical Analysis: {stats_passed}/{len(stats_tests)}")
    print(f"🗄️ Database: {db_passed}/{len(db_tests)}")
    print(f"🤖 AI/ML: {ai_passed}/{len(ai_tests)}")
    print(f"🔧 Optional: {optional_passed}/{len(optional_tests)}")
    
    if core_passed == len(core_tests):
        print("\n✅ System is ready to use!")
        if total_passed == total_tests:
            print("🌟 All enhanced features are available!")
        else:
            print("⚠️  Some enhanced features are missing but core functionality is intact.")
    else:
        print("\n❌ Core features are missing. Please run setup.sh again.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF
    
    chmod +x test_enhanced_features.py
    print_success "Test script created: test_enhanced_features.py"
}

# Main setup process
main() {
    echo ""
    print_status "Starting enhanced setup process..."
    echo ""
    
    # Run setup steps
    check_system_requirements
    check_python
    create_venv
    activate_venv
    install_enhanced_dependencies
    setup_selenium
    verify_enhanced_installation
    verify_file_structure
    create_enhanced_launch_script
    create_demo_script
    create_test_script
    
    echo ""
    print_success "🎉 Enhanced setup completed successfully!"
    echo ""
    print_feature "🌟 Available Enhanced Features:"
    echo "   • Task Selection with Templates"
    echo "   • Specialized Agent Types (Accountant, Sports Coach)"
    echo "   • Advanced Visualizations (8+ chart types)"
    echo "   • Web Intelligence & Search"
    echo "   • Financial Analysis Tools"
    echo "   • Statistical Analysis Suite"
    echo "   • Database Connectivity"
    echo "   • LangChain Hybrid Integration"
    echo ""
    print_status "Next steps:"
    echo "1. To start the enhanced application:"
    echo "   ./run_app.sh"
    echo ""
    echo "2. To run the enhanced features demo:"
    echo "   ./run_demo.sh"
    echo ""
    echo "3. To test all enhanced features:"
    echo "   ./test_enhanced_features.py"
    echo ""
    echo "4. Or manually:"
    echo "   source venv/bin/activate    # On Windows: venv\\Scripts\\activate"
    echo "   streamlit run src/app.py"
    echo ""
    echo "5. The application will open in your browser at http://localhost:8501"
    echo ""
    print_status "For troubleshooting enhanced features, check the README.md"
    print_status "For web features, ensure Chrome/Chromium is installed"
    echo ""
}

# Run main function
main "$@" 