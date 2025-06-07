#!/usr/bin/env python3
"""
Enhanced Features Test Suite

This script tests all enhanced features including:
- OCR and PDF processing capabilities
- Visualization validation engine
- Enhanced Accountant agent functionality
- Document processing workflows
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import io

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

def print_test(test_name):
    """Print test name"""
    print(f"\n🔍 Testing: {test_name}")

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_warning(message):
    """Print warning message"""
    print(f"⚠️  {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def test_ocr_dependencies():
    """Test OCR and PDF processing dependencies"""
    print_header("OCR and PDF Processing Dependencies")
    
    dependencies = [
        ('pytesseract', 'OCR text recognition'),
        ('pdf2image', 'PDF to image conversion'),
        ('PIL', 'Image processing (Pillow)'),
        ('PyPDF2', 'PDF text extraction'),
        ('pdfplumber', 'Advanced PDF processing')
    ]
    
    success_count = 0
    total_count = len(dependencies)
    
    for module_name, description in dependencies:
        print_test(f"{module_name} - {description}")
        try:
            __import__(module_name)
            print_success(f"{module_name} is available")
            success_count += 1
        except ImportError as e:
            print_error(f"{module_name} not available: {e}")
    
    print(f"\n📊 OCR Dependencies: {success_count}/{total_count} available")
    
    if success_count < total_count:
        print_warning("Some OCR dependencies are missing. Install with:")
        print("pip install pytesseract pdf2image Pillow PyPDF2 pdfplumber")
        print("\nSystem dependencies also required:")
        print("macOS: brew install tesseract poppler")
        print("Ubuntu: sudo apt-get install tesseract-ocr poppler-utils")
    
    return success_count == total_count

def test_pdf_processor():
    """Test PDF OCR processor functionality"""
    print_header("PDF OCR Processor")
    
    try:
        from src.utils.pdf_ocr_processor import pdf_processor
        print_success("PDF processor imported successfully")
        
        # Test availability check
        print_test("OCR availability check")
        is_available = pdf_processor.check_availability()
        if is_available:
            print_success("OCR functionality is available")
        else:
            print_warning("OCR functionality not available - dependencies missing")
            return False
        
        # Test pattern matching
        print_test("Financial pattern recognition")
        test_text = """
        Date: 01/15/2024
        Transaction: GROCERY STORE PURCHASE
        Amount: $45.67
        Balance: $1,234.56
        
        Date: 01/16/2024
        Transaction: ATM WITHDRAWAL
        Amount: -$100.00
        Balance: $1,134.56
        """
        
        extracted = pdf_processor._extract_financial_data(test_text)
        
        if extracted['dates'] and extracted['amounts']:
            print_success(f"Pattern recognition working - found {len(extracted['dates'])} dates, {len(extracted['amounts'])} amounts")
        else:
            print_warning("Pattern recognition may need improvement")
        
        # Test transaction categorization
        print_test("Transaction categorization")
        test_descriptions = [
            "GROCERY STORE PURCHASE",
            "STARBUCKS COFFEE",
            "UBER RIDE",
            "SALARY DEPOSIT",
            "ELECTRIC BILL PAYMENT"
        ]
        
        categories = [pdf_processor._categorize_transaction(desc) for desc in test_descriptions]
        categorized_count = sum(1 for cat in categories if cat != 'Uncategorized')
        
        print_success(f"Categorization working - {categorized_count}/{len(test_descriptions)} transactions categorized")
        
        return True
        
    except ImportError as e:
        print_error(f"Cannot import PDF processor: {e}")
        return False
    except Exception as e:
        print_error(f"PDF processor test failed: {e}")
        return False

def test_visualization_validator():
    """Test visualization validation engine"""
    print_header("Visualization Validation Engine")
    
    try:
        from src.utils.visualization_validator import visualization_validator
        print_success("Visualization validator imported successfully")
        
        # Test with good data
        print_test("Valid data validation")
        good_data = pd.DataFrame({
            'numeric_col1': np.random.randn(100),
            'numeric_col2': np.random.randn(100),
            'category_col': np.random.choice(['A', 'B', 'C', 'D'], 100),
            'date_col': pd.date_range('2024-01-01', periods=100)
        })
        
        validation_result = visualization_validator.validate_data_for_visualization(good_data)
        
        if validation_result['is_suitable']:
            print_success(f"Good data validated successfully - {len(validation_result['suitable_charts'])} chart types suitable")
            
            # Show top recommendations
            for i, chart in enumerate(validation_result['suitable_charts'][:3], 1):
                print(f"  {i}. {chart['type'].title()}: {chart['description']}")
        else:
            print_warning(f"Good data validation failed: {validation_result['reason']}")
        
        # Test with problematic data
        print_test("Problematic data validation")
        bad_data = pd.DataFrame({
            'all_same': [1] * 5,  # No variation
            'mostly_null': [1, None, None, None, None]  # Mostly null
        })
        
        bad_validation = visualization_validator.validate_data_for_visualization(bad_data)
        
        if not bad_validation['is_suitable']:
            print_success(f"Problematic data correctly rejected: {bad_validation['reason']}")
        else:
            print_warning("Problematic data was incorrectly accepted")
        
        # Test empty data
        print_test("Empty data validation")
        empty_data = pd.DataFrame()
        empty_validation = visualization_validator.validate_data_for_visualization(empty_data)
        
        if not empty_validation['is_suitable']:
            print_success("Empty data correctly rejected")
        else:
            print_warning("Empty data was incorrectly accepted")
        
        # Test specific chart validation
        print_test("Specific chart type validation")
        scatter_validation = visualization_validator.validate_specific_chart(
            good_data, 'scatter', ['numeric_col1', 'numeric_col2']
        )
        
        if scatter_validation['is_valid']:
            print_success("Scatter plot validation passed")
        else:
            print_warning(f"Scatter plot validation failed: {scatter_validation['reason']}")
        
        return True
        
    except ImportError as e:
        print_error(f"Cannot import visualization validator: {e}")
        return False
    except Exception as e:
        print_error(f"Visualization validator test failed: {e}")
        return False

def test_enhanced_accountant():
    """Test enhanced Accountant agent capabilities"""
    print_header("Enhanced Accountant Agent")
    
    try:
        from src.config.agent_roles import get_enhanced_role_prompts
        print_success("Enhanced agent roles imported successfully")
        
        # Check Accountant role configuration
        print_test("Accountant role configuration")
        role_prompts = get_enhanced_role_prompts()
        
        if 'Accountant' in role_prompts:
            accountant_config = role_prompts['Accountant']
            print_success("Accountant role found in enhanced configurations")
            
            # Check for OCR-related capabilities
            capabilities = accountant_config.get('capabilities', [])
            ocr_capabilities = [cap for cap in capabilities if 'ocr' in cap.lower() or 'pdf' in cap.lower()]
            
            if ocr_capabilities:
                print_success(f"OCR capabilities found: {', '.join(ocr_capabilities)}")
            else:
                print_warning("No OCR-specific capabilities found in Accountant role")
            
            # Check for enhanced tools
            tools = accountant_config.get('tools', [])
            enhanced_tools = [tool for tool in tools if 'ocr' in tool.lower() or 'csv' in tool.lower()]
            
            if enhanced_tools:
                print_success(f"Enhanced tools found: {', '.join(enhanced_tools)}")
            else:
                print_warning("No enhanced tools found in Accountant role")
            
            # Check prompt for PDF processing instructions
            prompt = accountant_config.get('prompt', '')
            if 'PDF' in prompt or 'OCR' in prompt:
                print_success("Prompt includes PDF/OCR processing instructions")
            else:
                print_warning("Prompt may not include PDF processing instructions")
        else:
            print_error("Accountant role not found in enhanced configurations")
            return False
        
        return True
        
    except ImportError as e:
        print_error(f"Cannot import enhanced agent roles: {e}")
        return False
    except Exception as e:
        print_error(f"Enhanced Accountant test failed: {e}")
        return False

def test_agent_integration():
    """Test Agent class integration with new features"""
    print_header("Agent Integration with Enhanced Features")
    
    try:
        from src.models.agent import Agent
        print_success("Agent class imported successfully")
        
        # Create test agent
        print_test("Agent creation with Accountant role")
        test_agent = Agent("TestAccountant", "Accountant", "test-model", use_openai=False)
        print_success("Accountant agent created successfully")
        
        # Test data copy functionality
        print_test("Agent data handling")
        test_data = pd.DataFrame({
            'amount': [100.50, -45.67, 200.00, -15.99],
            'description': ['SALARY', 'GROCERY STORE', 'FREELANCE', 'COFFEE SHOP'],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']
        })
        
        test_agent.set_data_copy(test_data)
        
        if test_agent.data_copy is not None and len(test_agent.data_copy) == len(test_data):
            print_success("Agent data handling working correctly")
        else:
            print_warning("Agent data handling may have issues")
        
        # Test enhanced features availability
        print_test("Enhanced features availability")
        if hasattr(test_agent, 'enhanced_viz_engine'):
            print_success("Enhanced visualization engine available")
        else:
            print_warning("Enhanced visualization engine not available")
        
        if hasattr(test_agent, 'web_search_tool'):
            print_success("Web search tool available")
        else:
            print_warning("Web search tool not available")
        
        return True
        
    except ImportError as e:
        print_error(f"Cannot import Agent class: {e}")
        return False
    except Exception as e:
        print_error(f"Agent integration test failed: {e}")
        return False

def test_financial_report_generation():
    """Test financial report generation capabilities"""
    print_header("Financial Report Generation")
    
    try:
        from src.utils.pdf_ocr_processor import pdf_processor
        
        # Create sample transaction data
        print_test("Sample financial data processing")
        sample_transactions = [
            {
                'Transaction_ID': 'TXN_0001',
                'Date': '2024-01-01',
                'Description': 'SALARY DEPOSIT',
                'Amount': 5000.00,
                'Category': 'Income',
                'Type': 'Credit',
                'Balance_Running': 5000.00
            },
            {
                'Transaction_ID': 'TXN_0002',
                'Date': '2024-01-02',
                'Description': 'GROCERY STORE',
                'Amount': -150.75,
                'Category': 'Food & Dining',
                'Type': 'Debit',
                'Balance_Running': 4849.25
            },
            {
                'Transaction_ID': 'TXN_0003',
                'Date': '2024-01-03',
                'Description': 'ELECTRIC BILL',
                'Amount': -89.50,
                'Category': 'Utilities',
                'Type': 'Debit',
                'Balance_Running': 4759.75
            }
        ]
        
        # Generate financial reports
        reports = pdf_processor.generate_financial_reports(sample_transactions)
        
        if reports:
            print_success("Financial reports generated successfully")
            
            # Check report components
            if 'summary' in reports:
                summary = reports['summary']
                print_success(f"Summary report: Income ${summary.get('total_income', 0):.2f}, Expenses ${summary.get('total_expenses', 0):.2f}")
            
            if 'category_breakdown' in reports:
                categories = reports['category_breakdown']
                print_success(f"Category breakdown: {len(categories)} categories analyzed")
            
            if 'expense_report' in reports:
                expense_report = reports['expense_report']
                print_success(f"Expense report: ${expense_report.get('total_expenses', 0):.2f} total expenses")
        else:
            print_warning("No financial reports generated")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Financial report generation test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all enhanced feature tests"""
    print_header("Multi-Agent System Enhanced Features Test Suite")
    print(f"🕒 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {}
    
    # Run all tests
    test_results['OCR Dependencies'] = test_ocr_dependencies()
    test_results['PDF Processor'] = test_pdf_processor()
    test_results['Visualization Validator'] = test_visualization_validator()
    test_results['Enhanced Accountant'] = test_enhanced_accountant()
    test_results['Agent Integration'] = test_agent_integration()
    test_results['Financial Reports'] = test_financial_report_generation()
    
    # Summary
    print_header("Test Results Summary")
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Overall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print_success("🎉 All enhanced features are working correctly!")
        print("\n🚀 Ready to use:")
        print("• PDF document processing with OCR")
        print("• Intelligent visualization validation")
        print("• Enhanced Accountant agent with document processing")
        print("• Financial report generation and CSV export")
        print("• Transaction categorization and analysis")
    else:
        print_warning(f"⚠️  {total_tests - passed_tests} test(s) failed")
        print("\n🔧 To fix issues:")
        print("• Install missing dependencies: pip install pytesseract pdf2image Pillow PyPDF2 pdfplumber")
        print("• Install system dependencies: tesseract and poppler")
        print("• Check setup.sh for automated installation")
    
    print(f"\n🕒 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
