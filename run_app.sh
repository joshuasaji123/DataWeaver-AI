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

