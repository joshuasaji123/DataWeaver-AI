"""
PDF OCR Processor for Financial Documents

This module provides comprehensive OCR and PDF processing capabilities
specifically designed for financial documents like bank statements.
"""

import os
import io
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import pytesseract
    from pdf2image import convert_from_bytes, convert_from_path
    from PIL import Image
    import PyPDF2
    import pdfplumber
    OCR_AVAILABLE = True
    logger.info("✅ OCR dependencies loaded successfully")
except ImportError as e:
    OCR_AVAILABLE = False
    logger.warning(f"⚠️ OCR dependencies not available: {e}")

class PDFOCRProcessor:
    """
    Advanced PDF and OCR processor for financial documents
    """
    
    def __init__(self):
        """Initialize the PDF OCR processor"""
        self.ocr_available = OCR_AVAILABLE
        self.supported_formats = ['.pdf']
        
        # Common financial patterns for extraction
        self.patterns = {
            'date': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or M/D/YY
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD
                r'\b\d{1,2}\s+\w{3}\s+\d{4}\b',       # DD MMM YYYY
                r'\b\w{3}\s+\d{1,2},?\s+\d{4}\b'      # MMM DD, YYYY
            ],
            'amount': [
                r'\$?\s*-?\d{1,3}(?:,\d{3})*\.?\d{0,2}',  # Currency amounts
                r'-?\d+\.\d{2}',  # Decimal amounts
                r'\(\$?\d{1,3}(?:,\d{3})*\.?\d{0,2}\)'  # Negative amounts in parentheses
            ],
            'transaction_id': [
                r'\b[A-Z0-9]{6,}\b',  # Transaction IDs
                r'\b\d{10,}\b'  # Long numeric IDs
            ],
            'account_number': [
                r'(?:Account|Acct).*?(\d{4,})',  # Account numbers
                r'xxxx-?(\d{4})'  # Masked account numbers
            ]
        }
        
        # Keywords for categorizing transactions
        self.expense_categories = {
            'Food & Dining': ['restaurant', 'food', 'dining', 'cafe', 'pizza', 'burger', 'starbucks', 'coffee'],
            'Transportation': ['gas', 'fuel', 'uber', 'lyft', 'taxi', 'parking', 'metro', 'bus', 'train'],
            'Shopping': ['amazon', 'walmart', 'target', 'store', 'mall', 'shop', 'retail'],
            'Utilities': ['electric', 'gas company', 'water', 'internet', 'phone', 'cable', 'utility'],
            'Healthcare': ['hospital', 'doctor', 'pharmacy', 'medical', 'dental', 'vision', 'health'],
            'Entertainment': ['movie', 'netflix', 'spotify', 'game', 'entertainment', 'theater'],
            'Banking': ['fee', 'interest', 'transfer', 'withdrawal', 'deposit', 'atm'],
            'Insurance': ['insurance', 'premium', 'policy'],
            'Income': ['salary', 'payroll', 'deposit', 'payment', 'refund', 'dividend']
        }
    
    def check_availability(self) -> bool:
        """Check if OCR functionality is available"""
        return self.ocr_available
    
    def process_pdf(self, file_content: bytes, filename: str) -> Dict:
        """
        Process a PDF file and extract financial data
        
        Args:
            file_content: PDF file content as bytes
            filename: Name of the uploaded file
            
        Returns:
            Dictionary containing extracted data and processed results
        """
        if not self.ocr_available:
            return {
                'success': False,
                'error': 'OCR dependencies not available. Please install: pytesseract, pdf2image, Pillow',
                'data': None
            }
        
        try:
            logger.info(f"Processing PDF: {filename}")
            
            # Try text extraction first (faster for text-based PDFs)
            text_data = self._extract_text_from_pdf(file_content)
            
            if not text_data or len(text_data.strip()) < 100:
                # Fall back to OCR for image-based PDFs
                logger.info("PDF appears to be image-based, using OCR...")
                text_data = self._extract_text_with_ocr(file_content)
            
            if not text_data:
                return {
                    'success': False,
                    'error': 'Could not extract text from PDF',
                    'data': None
                }
            
            # Extract financial data from text
            extracted_data = self._extract_financial_data(text_data)
            
            # Convert to structured format
            structured_data = self._structure_financial_data(extracted_data)
            
            # Generate CSV-ready data
            csv_data = self._generate_csv_data(structured_data)
            
            return {
                'success': True,
                'raw_text': text_data,
                'extracted_data': extracted_data,
                'structured_data': structured_data,
                'csv_data': csv_data,
                'transactions_count': len(csv_data) if csv_data else 0,
                'processing_method': 'text_extraction' if len(text_data.strip()) >= 100 else 'ocr'
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {
                'success': False,
                'error': f'Error processing PDF: {str(e)}',
                'data': None
            }
    
    def _extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF using PyPDF2 and pdfplumber"""
        text = ""
        
        try:
            # Try pdfplumber first (better for complex layouts)
            with io.BytesIO(file_content) as pdf_file:
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            
            if len(text.strip()) > 50:
                return text
                
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        try:
            # Fall back to PyPDF2
            with io.BytesIO(file_content) as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                    
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
        
        return text
    
    def _extract_text_with_ocr(self, file_content: bytes) -> str:
        """Extract text using OCR for image-based PDFs"""
        try:
            # Convert PDF to images
            images = convert_from_bytes(file_content, dpi=300)
            
            text = ""
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1} with OCR...")
                
                # Apply OCR to each page
                page_text = pytesseract.image_to_string(image, lang='eng')
                text += page_text + "\n"
            
            return text
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def _extract_financial_data(self, text: str) -> Dict:
        """Extract financial data patterns from text"""
        extracted = {
            'dates': [],
            'amounts': [],
            'descriptions': [],
            'transaction_ids': [],
            'account_info': []
        }
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract dates
            for pattern in self.patterns['date']:
                dates = re.findall(pattern, line)
                extracted['dates'].extend(dates)
            
            # Extract amounts
            for pattern in self.patterns['amount']:
                amounts = re.findall(pattern, line)
                extracted['amounts'].extend(amounts)
            
            # Extract transaction IDs
            for pattern in self.patterns['transaction_id']:
                ids = re.findall(pattern, line)
                extracted['transaction_ids'].extend(ids)
            
            # Store line as potential description
            if any(keyword in line.lower() for keyword in ['transaction', 'payment', 'deposit', 'withdrawal']):
                extracted['descriptions'].append(line)
        
        # Extract account information
        for pattern in self.patterns['account_number']:
            accounts = re.findall(pattern, text, re.IGNORECASE)
            extracted['account_info'].extend(accounts)
        
        return extracted
    
    def _structure_financial_data(self, extracted: Dict) -> List[Dict]:
        """Structure extracted data into transaction records"""
        transactions = []
        
        # Simple heuristic: try to match dates with amounts and descriptions
        dates = extracted['dates']
        amounts = extracted['amounts']
        descriptions = extracted['descriptions']
        
        # Parse and standardize dates
        parsed_dates = []
        for date_str in dates:
            try:
                # Try different date formats
                for fmt in ['%m/%d/%Y', '%m/%d/%y', '%Y/%m/%d', '%d %b %Y', '%b %d, %Y']:
                    try:
                        parsed_date = datetime.strptime(date_str.replace(',', ''), fmt)
                        parsed_dates.append((date_str, parsed_date))
                        break
                    except ValueError:
                        continue
            except:
                continue
        
        # Parse and clean amounts
        cleaned_amounts = []
        for amount_str in amounts:
            try:
                # Remove currency symbols and commas
                clean_amount = re.sub(r'[,$]', '', amount_str)
                
                # Handle negative amounts in parentheses
                if '(' in amount_str and ')' in amount_str:
                    clean_amount = '-' + clean_amount.replace('(', '').replace(')', '')
                
                amount_val = float(clean_amount)
                cleaned_amounts.append((amount_str, amount_val))
            except:
                continue
        
        # Create transaction records
        max_transactions = min(len(parsed_dates), len(cleaned_amounts))
        
        for i in range(max_transactions):
            transaction = {
                'date': parsed_dates[i][1].strftime('%Y-%m-%d'),
                'original_date': parsed_dates[i][0],
                'amount': cleaned_amounts[i][1],
                'original_amount': cleaned_amounts[i][0],
                'description': descriptions[i] if i < len(descriptions) else '',
                'category': self._categorize_transaction(descriptions[i] if i < len(descriptions) else ''),
                'transaction_type': 'Credit' if cleaned_amounts[i][1] > 0 else 'Debit'
            }
            transactions.append(transaction)
        
        # Sort by date
        transactions.sort(key=lambda x: x['date'])
        
        return transactions
    
    def _categorize_transaction(self, description: str) -> str:
        """Categorize transaction based on description"""
        description_lower = description.lower()
        
        for category, keywords in self.expense_categories.items():
            if any(keyword in description_lower for keyword in keywords):
                return category
        
        return 'Uncategorized'
    
    def _generate_csv_data(self, transactions: List[Dict]) -> List[Dict]:
        """Generate CSV-ready data from transactions"""
        if not transactions:
            return []
        
        csv_data = []
        
        for i, transaction in enumerate(transactions, 1):
            csv_row = {
                'Transaction_ID': f"TXN_{i:04d}",
                'Date': transaction['date'],
                'Description': transaction['description'][:100],  # Limit description length
                'Amount': transaction['amount'],
                'Category': transaction['category'],
                'Type': transaction['transaction_type'],
                'Balance_Running': self._calculate_running_balance(transactions, i-1)
            }
            csv_data.append(csv_row)
        
        return csv_data
    
    def _calculate_running_balance(self, transactions: List[Dict], current_index: int) -> float:
        """Calculate running balance up to current transaction"""
        balance = 0.0
        for i in range(current_index + 1):
            balance += transactions[i]['amount']
        return round(balance, 2)
    
    def generate_financial_reports(self, csv_data: List[Dict]) -> Dict:
        """Generate financial reports from transaction data"""
        if not csv_data:
            return {}
        
        df = pd.DataFrame(csv_data)
        
        # Convert date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        reports = {
            'summary': self._generate_summary_report(df),
            'category_breakdown': self._generate_category_report(df),
            'monthly_analysis': self._generate_monthly_report(df),
            'expense_report': self._generate_expense_report(df),
            'income_analysis': self._generate_income_analysis(df)
        }
        
        return reports
    
    def _generate_summary_report(self, df: pd.DataFrame) -> Dict:
        """Generate financial summary report"""
        total_income = df[df['Amount'] > 0]['Amount'].sum()
        total_expenses = abs(df[df['Amount'] < 0]['Amount'].sum())
        net_cash_flow = total_income - total_expenses
        
        return {
            'period_start': df['Date'].min().strftime('%Y-%m-%d'),
            'period_end': df['Date'].max().strftime('%Y-%m-%d'),
            'total_income': round(total_income, 2),
            'total_expenses': round(total_expenses, 2),
            'net_cash_flow': round(net_cash_flow, 2),
            'transaction_count': len(df),
            'average_transaction': round(df['Amount'].mean(), 2)
        }
    
    def _generate_category_report(self, df: pd.DataFrame) -> Dict:
        """Generate category breakdown report"""
        category_summary = df.groupby('Category').agg({
            'Amount': ['sum', 'count', 'mean']
        }).round(2)
        
        category_data = {}
        for category in category_summary.index:
            category_data[category] = {
                'total_amount': category_summary.loc[category, ('Amount', 'sum')],
                'transaction_count': category_summary.loc[category, ('Amount', 'count')],
                'average_amount': category_summary.loc[category, ('Amount', 'mean')]
            }
        
        return category_data
    
    def _generate_monthly_report(self, df: pd.DataFrame) -> Dict:
        """Generate monthly analysis report"""
        df['Month'] = df['Date'].dt.to_period('M')
        
        monthly_data = df.groupby('Month').agg({
            'Amount': ['sum', 'count']
        }).round(2)
        
        monthly_report = {}
        for month in monthly_data.index:
            monthly_report[str(month)] = {
                'total_amount': monthly_data.loc[month, ('Amount', 'sum')],
                'transaction_count': monthly_data.loc[month, ('Amount', 'count')]
            }
        
        return monthly_report
    
    def _generate_expense_report(self, df: pd.DataFrame) -> Dict:
        """Generate detailed expense report"""
        expenses_df = df[df['Amount'] < 0].copy()
        expenses_df['Amount'] = abs(expenses_df['Amount'])
        
        if expenses_df.empty:
            return {'total_expenses': 0, 'categories': {}}
        
        category_expenses = expenses_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
        
        return {
            'total_expenses': round(expenses_df['Amount'].sum(), 2),
            'categories': category_expenses.round(2).to_dict(),
            'largest_expense': {
                'amount': round(expenses_df['Amount'].max(), 2),
                'description': expenses_df.loc[expenses_df['Amount'].idxmax(), 'Description']
            }
        }
    
    def _generate_income_analysis(self, df: pd.DataFrame) -> Dict:
        """Generate income analysis report"""
        income_df = df[df['Amount'] > 0].copy()
        
        if income_df.empty:
            return {'total_income': 0, 'sources': {}}
        
        income_sources = income_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
        
        return {
            'total_income': round(income_df['Amount'].sum(), 2),
            'sources': income_sources.round(2).to_dict(),
            'largest_income': {
                'amount': round(income_df['Amount'].max(), 2),
                'description': income_df.loc[income_df['Amount'].idxmax(), 'Description']
            }
        }


# Global instance for easy access
pdf_processor = PDFOCRProcessor() 