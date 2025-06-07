# 🌟 DataWeaver-AI
### *Intelligent Multi-Agent Data Analysis with PDF Processing & OCR*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red.svg)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/Powered%20by-OpenAI-green.svg)](https://openai.com/)

**An intelligent multi-agent system that weaves insights from your data using AI-powered analysis, PDF processing, and collaborative workflows.**

DataWeaver-AI employs specialized AI agents that work together to analyze data, process documents, and generate comprehensive insights. From bank statements to complex datasets, our agents transform raw information into actionable intelligence.

## ✨ Key Features

- 🤖 **Multi-Agent Intelligence**: Specialized AI agents (Business Analyst, Accountant, Data Scientist, etc.)
- 📄 **PDF Processing**: OCR extraction from bank statements, invoices, and financial documents  
- 📊 **Smart Visualizations**: Intelligent chart creation with validation to prevent empty graphs
- 💰 **Financial Analysis**: Automated expense categorization, P&L generation, tax preparation
- 🔍 **Data Validation**: Pre-visualization validation ensuring meaningful insights
- 🌐 **Web Intelligence**: Real-time data collection and sentiment analysis
- 📈 **Advanced Analytics**: Statistical testing, ML integration, predictive modeling

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/joshuasaji123/DataWeaver-AI.git
cd DataWeaver-AI
chmod +x setup.sh
./setup.sh

# Launch application
./run_app.sh
```

### Usage
1. **Upload Data**: CSV files or PDF documents (bank statements, invoices)
2. **Configure Agents**: Choose specialized roles and AI models
3. **Select Mode**: Supervised, Unsupervised, or Autonomous execution
4. **Watch Magic**: Agents analyze, clean, visualize, and summarize your data
5. **Export Results**: Download insights, reports, and processed data

## 🎯 Agent Workflow

```
🔍 ANALYZE → 🔧 CLEAN → 📊 VISUALIZE → 📋 SUMMARIZE → 🔄 REPEAT → ✅ END
```

Each agent follows an intelligent workflow:
- **Analyze**: Deep data exploration with statistical analysis
- **Clean**: Automated data quality improvement
- **Visualize**: Smart chart creation based on data characteristics
- **Summarize**: Executive-level insights with recommendations
- **Repeat**: AI-driven decisions about additional analysis cycles
- **End**: Autonomous task completion

## 🤖 Specialized Agents

### 📊 Accountant Agent
- **PDF Processing**: OCR extraction from bank statements and financial documents
- **Transaction Analysis**: Automatic categorization and expense tracking
- **Financial Reports**: P&L statements, expense reports, cash flow analysis
- **Tax Preparation**: Deductible expense identification and summaries

### 💼 Business Analyst
- **Market Intelligence**: Competitive analysis and growth opportunities
- **Customer Analytics**: Segmentation, lifetime value, behavior patterns
- **Performance Metrics**: ROI, KPIs, conversion analysis
- **Strategic Planning**: SWOT analysis and business intelligence

### 📈 Data Scientist
- **Machine Learning**: Predictive modeling and advanced analytics
- **Statistical Analysis**: Hypothesis testing and significance analysis
- **Feature Engineering**: Automated feature selection and optimization
- **Model Validation**: Cross-validation and performance evaluation

## 🛠️ Installation

### Prerequisites
- Python 3.8+ (recommended: 3.9-3.11)
- 8GB+ RAM (varies by model selection)
- 2GB+ free storage

### Automated Setup
```bash
# Validate system (optional)
python validate_setup.py

# Run enhanced setup
chmod +x setup.sh
./setup.sh

# Test installation
python test_enhanced_features.py
```

### OCR System Dependencies (Optional)
For PDF processing capabilities:

**macOS:**
```bash
brew install tesseract poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr poppler-utils
```

## 🔧 Configuration

### API Setup
Choose your AI provider:

**OpenAI (Cloud):**
1. Get API key from [OpenAI](https://platform.openai.com/api-keys)
2. Select "OpenAI" in the app
3. Enter your API key
4. Choose model: `gpt-4`, `gpt-3.5-turbo`

**Ollama (Local):**
1. Install [Ollama](https://ollama.ai)
2. Pull a model: `ollama pull llama2`
3. Select "Ollama" in the app
4. Choose your model

## 📁 Project Structure

```
DataWeaver-AI/
├── src/
│   ├── app.py                           # Main Streamlit application
│   ├── models/
│   │   ├── agent.py                     # Core Agent class with AI capabilities
│   │   └── enhanced_agent.py            # LangChain-enhanced Agent
│   ├── utils/
│   │   ├── model_manager.py             # Model management and recommendations
│   │   ├── pdf_ocr_processor.py         # PDF processing with OCR
│   │   └── visualization_validator.py   # Intelligent chart validation
│   ├── config/
│   │   ├── prompts.py                   # Agent prompts and configurations
│   │   └── agent_roles.py               # Enhanced role definitions
│   └── ui/
│       └── components.py                # Modular UI components
├── test_enhanced_features.py            # Comprehensive feature tests
├── validate_setup.py                    # System validation
├── requirements.txt                     # Python dependencies
├── setup.sh                             # Automated setup script
├── run_app.sh                           # Application launcher
└── README.md                            # This documentation
```

## 🎮 Execution Modes

### 🎯 Supervised Mode
- Manual control with user approval at each step
- Interactive feedback and guidance
- Perfect for learning and understanding agent decisions

### 🤖 Unsupervised Mode
- AI self-feedback with minimal user intervention
- Automated workflow progression with built-in delays
- Live progress tracking and performance metrics

### 🚁 Autonomous Mode
- Complete agent autonomy with independent decision-making
- Real-time monitoring with emergency controls
- Demonstrates true agentic AI behavior

## 🔍 Enhanced Features

### PDF Intelligence
- **Bank Statement Processing**: Automatic transaction extraction
- **Invoice Analysis**: OCR for expense tracking
- **Financial Document Recognition**: Multi-format support
- **Data Validation**: Intelligent verification with error detection

### Smart Visualizations
- **Data Validation**: Pre-visualization suitability checks
- **Empty Graph Prevention**: Ensures meaningful charts only
- **Chart Recommendations**: AI-driven optimal visualization selection
- **Interactive Dashboards**: Real-time data exploration

### Web Intelligence
- **Smart Search**: Role-based search strategies
- **Sentiment Analysis**: Automatic news sentiment scoring
- **Market Data**: Real-time financial and sports data integration
- **Competitive Intelligence**: Basic market positioning insights

## 🛠️ Troubleshooting

### Common Issues

**Empty Visualizations:**
- Ensure data has numeric/categorical columns
- Check for NaN values
- Verify data quality

**OCR Processing:**
- Install system dependencies: `tesseract` and `poppler`
- Ensure PDF files aren't password-protected
- Check file size limits (< 50MB recommended)

**Missing Dependencies:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Test specific features
python test_enhanced_features.py
```

## 📊 Performance

- **Large Datasets**: Automatic sampling for visualization performance
- **Memory Usage**: Built-in monitoring and optimization
- **Model Selection**: Recommendations based on system resources
- **Enhanced Features**: Optional advanced capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Powered by [OpenAI](https://openai.com/) and [Ollama](https://ollama.ai/) for AI capabilities
- Enhanced with [LangChain](https://langchain.com/) for robust agent frameworks
- OCR capabilities provided by [Tesseract](https://github.com/tesseract-ocr/tesseract)

---

**⭐ Star this repository if you find DataWeaver-AI helpful!** 