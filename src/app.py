"""
Main application module for the Multi-Agent Data Analysis System.

This module provides the Streamlit-based user interface and coordinates the interaction
between different components of the system. It implements a sophisticated multi-agent
workflow for data analysis that follows the sequence:

WORKFLOW: Analyze → Data Cleaning (if needed) → Visualize → Summarize → Repeat → End

The system supports three operation modes:
1. Supervised: Manual control with user feedback at each step
2. Unsupervised: Automated with AI self-feedback and delays
3. Headless (Autonomous): Fully autonomous AI agents with real-time monitoring

Key Features:
- Multi-agent collaboration with role-based analysis
- Real-time performance monitoring and timing
- Interactive visualizations with Plotly charts
- Data isolation (each agent gets its own copy)
- Comprehensive analysis history and learning points
- Support for both OpenAI and Ollama models
- Live progress tracking and workflow visualization

Architecture:
- Agent: Core analysis engine with specific roles and decision-making
- ModelManager: Handles model requirements and recommendations
- UI Components: Modular display components for different data types
- Config: Role-based prompts and system configuration
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import openai
import sys
from pathlib import Path

# Add the project root directory to Python path
# This allows importing custom modules from the src directory
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import custom modules for the multi-agent system
from src.models.agent import Agent  # Core agent class with analysis capabilities

# Import enhanced agent for hybrid LangChain integration
try:
    from src.models.enhanced_agent import EnhancedAgent, create_enhanced_agent
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"LangChain integration not available: {e}")

from src.utils.model_manager import ModelManager  # Handles model requirements and selection
from src.config.prompts import get_role_prompts  # Predefined role-based analysis prompts
from src.ui.components import (  # Modular UI components for different display needs
    display_agent_thoughts,        # Shows agent's thought process and reasoning
    display_conversation,          # Displays conversation history between user and agents
    display_agent_status,          # Shows agent goals, progress, and learning points
    display_analysis_results,      # Comprehensive analysis results display
    display_model_recommendations, # Shows recommended models based on role/requirements
    display_system_resources,      # Displays available system resources (RAM, CPU)
    display_model_requirements,    # Shows model memory requirements and compatibility
    display_cycle_timing,          # Live timing information and performance metrics
    display_data_manipulations,    # Shows data cleaning and manipulation history
    display_agent_visualizations,  # Displays interactive charts created by agents
    display_langchain_stats,       # Shows LangChain integration statistics and performance
    display_hybrid_agent_comparison # Displays comparison between standard and enhanced agents
)

def main():
    """
    Main function to run the Streamlit application.
    
    This function orchestrates the entire multi-agent data analysis system:
    1. Sets up the user interface and configuration options
    2. Manages API keys and model selection
    3. Handles file upload and data preview
    4. Configures agents with roles, models, and custom prompts
    5. Executes analysis in different modes (Supervised/Unsupervised/Headless)
    6. Coordinates agent interactions and workflow management
    7. Displays results, timing, and comprehensive analysis information
    8. Manages post-analysis feedback and agent responses
    
    The function implements a sophisticated workflow management system that ensures
    agents follow the proper sequence: Analyze → Clean → Visualize → Summarize → Repeat
    """
    
    # ==================================================================================
    # SECTION 1: APPLICATION SETUP AND CONFIGURATION
    # ==================================================================================
    
    # Set up the main application title
    st.title("Multi-Agent Data Analysis System")
    
    # Initialize model manager for handling model-related operations
    # This manages model requirements, recommendations, and system compatibility
    model_manager = ModelManager()
    
    # ==================================================================================
    # SECTION 2: API CONFIGURATION AND MODEL SETUP
    # ==================================================================================
    
    # OpenAI API Key configuration in sidebar
    # Users can choose between OpenAI (cloud) or Ollama (local) models
    st.sidebar.header("API Configuration")
    use_openai = st.sidebar.checkbox("Use OpenAI API", value=False)
    
    if use_openai:
        # OpenAI API key input with secure password masking
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if openai_api_key:
            openai.api_key = openai_api_key
            st.sidebar.success("OpenAI API Key configured!")
        else:
            st.sidebar.warning("Please enter your OpenAI API Key")
    
    # ==================================================================================
    # SECTION 2.5: HYBRID MODE CONFIGURATION (LANGCHAIN INTEGRATION)
    # ==================================================================================
    
    st.sidebar.header("🔬 Hybrid Mode (LangChain Integration)")
    
    if LANGCHAIN_AVAILABLE:
        # Allow users to choose between standard and enhanced agents
        use_enhanced_agents = st.sidebar.checkbox(
            "🚀 Enable LangChain Enhanced Agents", 
            value=False,
            help="Use LangChain framework for improved agent capabilities, error handling, and tool integration"
        )
        
        if use_enhanced_agents:
            st.sidebar.success("✅ Enhanced agents enabled with LangChain integration!")
            st.sidebar.info("""
            **Enhanced Features:**
            - Improved error handling and retry mechanisms
            - Advanced tool integration with ReAct framework
            - Better prompt management and memory systems
            - Production-ready monitoring and observability
            - Fallback to standard agents if needed
            """)
        else:
            st.sidebar.info("Using standard agents. Enable hybrid mode for LangChain integration.")
    else:
        use_enhanced_agents = False
        st.sidebar.warning("⚠️ LangChain not available. Install requirements: `pip install langchain>=0.1.0`")
    
    # Display system resources and model requirements in sidebar
    # This helps users understand what models their system can run
    resources = model_manager.get_system_resources()
    display_system_resources(resources, st.sidebar)
    
    model_requirements = model_manager.get_model_requirements()
    display_model_requirements(model_requirements, resources, st.sidebar)
    
    # ==================================================================================
    # SECTION 3: DATA UPLOAD AND PREVIEW
    # ==================================================================================
    
    # File upload section for CSV and PDF data input
    # Support for both structured CSV data and unstructured PDF documents (like bank statements)
    st.subheader("📁 Data Upload")
    
    # File type selection
    file_type = st.radio(
        "Select file type to upload:",
        ["CSV Data", "PDF Document (OCR)"],
        help="Choose CSV for structured data or PDF for documents like bank statements that need OCR processing"
    )
    
    if file_type == "CSV Data":
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        
        if uploaded_file is not None:
            # Load and display data preview to give users confidence in their upload
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(data.head())  # Show first 5 rows for quick verification
                
                # Validate data for visualization
                from src.utils.visualization_validator import visualization_validator
                viz_validation = visualization_validator.validate_data_for_visualization(data)
                
                # Display visualization validation results
                if viz_validation['is_suitable']:
                    st.success("✅ Data is suitable for visualization!")
                    with st.expander("📊 Visualization Recommendations", expanded=False):
                        st.write("**Recommended Chart Types:**")
                        for chart in viz_validation['suitable_charts'][:3]:
                            st.write(f"• **{chart['type'].title()}**: {chart['description']}")
                        
                        st.write("**Data Analysis:**")
                        summary = viz_validation['data_summary']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Rows", summary['row_count'])
                        with col2:
                            st.metric("Total Columns", summary['column_count'])
                        with col3:
                            st.metric("Numeric Columns", len(summary['numeric_columns']))
                        
                        if viz_validation['recommendations']:
                            st.write("**Recommendations:**")
                            for rec in viz_validation['recommendations']:
                                st.write(f"• {rec}")
                else:
                    st.warning(f"⚠️ Visualization limitations: {viz_validation['reason']}")
                    if viz_validation['recommendations']:
                        st.write("**Suggestions:**")
                        for rec in viz_validation['recommendations']:
                            st.write(f"• {rec}")
                
                # Store CSV data in session state for agent analysis
                st.session_state['csv_data'] = data
                st.session_state['viz_validation'] = viz_validation
                st.session_state['uploaded_filename'] = uploaded_file.name
                
                # Option to download processed data
                csv_string = data.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV Data",
                    data=csv_string,
                    file_name=uploaded_file.name,
                    mime="text/csv"
                )
                
                st.info("✅ Data ready for agent analysis! Configure agents below to start the analysis.")
                
            except Exception as e:
                st.error(f"Error loading CSV file: {str(e)}")
                return
    
    else:  # PDF Document processing
        uploaded_file = st.file_uploader(
            "Upload your PDF document (Bank Statements, Financial Reports, etc.)",
            type=['pdf'],
            help="Upload PDF documents like bank statements for OCR processing and financial analysis"
        )
        
        if uploaded_file is not None:
            st.write("📄 PDF Document Uploaded:")
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            
            # Import OCR processor
            from src.utils.pdf_ocr_processor import pdf_processor
            
            # Check OCR availability
            if not pdf_processor.check_availability():
                st.error("""
                🚫 **OCR Dependencies Not Available**
                
                To process PDF documents, please install the required dependencies:
                ```bash
                pip install pytesseract pdf2image Pillow PyPDF2 pdfplumber
                ```
                
                **Additional System Requirements:**
                - **Tesseract OCR**: Install from https://github.com/tesseract-ocr/tesseract
                - **Poppler**: Required for pdf2image conversion
                
                **Installation Instructions:**
                - **macOS**: `brew install tesseract poppler`
                - **Ubuntu**: `sudo apt-get install tesseract-ocr poppler-utils`
                - **Windows**: Download from official sites or use conda
                """)
                return
            
            # Process PDF button
            if st.button("🔍 Process PDF with OCR", type="primary"):
                with st.spinner("Processing PDF document... This may take a few moments."):
                    # Read file content
                    file_content = uploaded_file.read()
                    
                    # Process with OCR
                    result = pdf_processor.process_pdf(file_content, uploaded_file.name)
                    
                    if result['success']:
                        st.success(f"✅ Successfully processed PDF! Found {result['transactions_count']} transactions.")
                        
                        # Display processing information
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Processing Method", result['processing_method'].title())
                        with col2:
                            st.metric("Transactions Found", result['transactions_count'])
                        
                        # Convert to DataFrame for analysis
                        if result['csv_data']:
                            data = pd.DataFrame(result['csv_data'])
                            
                            # Display processed data preview
                            st.write("**📊 Processed Transaction Data:**")
                            st.dataframe(data.head(10))
                            
                            # Generate financial reports
                            reports = pdf_processor.generate_financial_reports(result['csv_data'])
                            
                            if reports:
                                st.write("**📈 Financial Analysis Summary:**")
                                
                                # Summary metrics
                                if 'summary' in reports:
                                    summary = reports['summary']
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Total Income", f"${summary.get('total_income', 0):.2f}")
                                    with col2:
                                        st.metric("Total Expenses", f"${summary.get('total_expenses', 0):.2f}")
                                    with col3:
                                        st.metric("Net Cash Flow", f"${summary.get('net_cash_flow', 0):.2f}")
                                    with col4:
                                        st.metric("Avg Transaction", f"${summary.get('average_transaction', 0):.2f}")
                                
                                # Category breakdown
                                if 'expense_report' in reports and reports['expense_report']['categories']:
                                    st.write("**💳 Expense Categories:**")
                                    categories_df = pd.DataFrame([
                                        {'Category': cat, 'Amount': amt}
                                        for cat, amt in reports['expense_report']['categories'].items()
                                    ]).sort_values('Amount', ascending=False)
                                    st.dataframe(categories_df)
                            
                            # Option to download CSV
                            csv_string = data.to_csv(index=False)
                            st.download_button(
                                label="📥 Download as CSV",
                                data=csv_string,
                                file_name=f"processed_{uploaded_file.name.replace('.pdf', '.csv')}",
                                mime="text/csv"
                            )
                            
                            # Store OCR results in session state for agent analysis
                            st.session_state['ocr_result'] = result
                            st.session_state['reports'] = reports
                            
                        else:
                            st.warning("⚠️ No structured transaction data could be extracted from the PDF.")
                            
                            # Display raw text for manual review
                            if result.get('raw_text'):
                                with st.expander("📄 View Extracted Text", expanded=False):
                                    st.text_area("Raw OCR Text", result['raw_text'], height=300)
                    
                    else:
                        st.error(f"❌ Failed to process PDF: {result['error']}")
                        return
    
    # ==================================================================================
    # SECTION 4: AGENT CONFIGURATION AND SETUP
    # ==================================================================================
    
        st.subheader("Configure Agents")
    
    # Allow users to configure multiple agents for collaborative analysis
    num_agents = st.number_input("Number of Agents", min_value=1, max_value=5, value=1)
    
    agents = []  # List to store configured agent instances
    
    # Configure each agent individually
    for i in range(num_agents):
        st.write(f"Agent {i+1} Configuration")
        
        # Agent name input - helps identify agents in collaborative scenarios
        agent_name = st.text_input(f"Agent {i+1} Name", f"Agent_{i+1}")
        
        # Role selection with predefined options
        # Roles determine the agent's analysis approach and focus areas
        role_options = list(get_role_prompts().keys())
        selected_role = st.selectbox(
            f"Agent {i+1} Role",
            role_options,
            help="Select a predefined role or create a custom one"
        )
        
        # Custom role input if selected
        # Allows users to define specialized analysis roles
        if selected_role == "Custom":
            agent_role = st.text_input(
                f"Custom Role for Agent {i+1}",
                placeholder="e.g., Market Researcher, Financial Analyst, etc."
            )
        else:
            agent_role = selected_role
        
        # Get and display model recommendations based on role
        # Different roles may benefit from different model capabilities
        role_info = get_role_prompts()[selected_role]
        goals = role_info["sub_goals"] if selected_role != "Custom" else []
        model_recommendations = model_manager.get_model_recommendations(agent_role, goals)
        
        display_model_recommendations(model_recommendations, use_openai, st)
        
        # Model selection based on available options and system compatibility
        if use_openai:
            model_options = [m["name"] for m in model_recommendations["openai_models"]]
        else:
            available_models = model_manager.get_available_models()
            model_options = [m['name'] for m in available_models] if available_models else [m["name"] for m in model_recommendations["ollama_models"]]
        
        # Create formatted model options with memory requirements for user guidance
        model_options_formatted = []
        for model_name in model_options:
            if model_name in model_requirements:
                size = model_requirements[model_name]['size_gb']
                model_options_formatted.append(f"{model_name} ({size:.2f} GB)")
            else:
                model_info = next((m for m in available_models if m['name'] == model_name), None)
                size = model_info['size_gb'] if model_info else 0.0
                model_options_formatted.append(f"{model_name} ({size:.2f} GB)")
        
        # Model selection dropdown with resource information
        selected_model_formatted = st.selectbox(
            f"Model for Agent {i+1}",
            model_options_formatted,
            help="Select a model based on your system resources and requirements"
        )
        
        # Extract model name from formatted string (remove size information)
        selected_model = selected_model_formatted.split(" (")[0]
        
        # Show model requirements warning if system doesn't meet minimum requirements
        if selected_model in model_requirements:
            reqs = model_requirements[selected_model]
            if resources['memory_gb'] < reqs['min_memory_gb']:
                st.warning(f"⚠️ Warning: This model requires at least {reqs['min_memory_gb']}GB of memory")
        
        # ==================================================================================
        # SECTION 5: PROMPT CUSTOMIZATION
        # ==================================================================================
        
        # Prompt customization section for advanced users
        # Allows fine-tuning of agent behavior and analysis focus
        st.write("### Prompt Customization")
        with st.expander(f"Customize Prompt for Agent {i+1}", expanded=False):
            st.write("""
            Customize the analysis prompt for this agent. Use the following placeholders:
            - {role}: The agent's role
            - {context}: Additional context for the analysis
            - {data_summary}: Summary statistics of the data
            """)
            
            # Get default prompt for selected role
            default_prompt = role_info["prompt"]
            
            # Allow custom prompt modification
            custom_prompt = st.text_area(
                f"Analysis Prompt Template for Agent {i+1}",
                value=default_prompt,
                height=200,
                help="Modify the prompt template to customize the analysis approach"
            )
            
            # Additional context input for specific analysis requirements
            analysis_context = st.text_area(
                f"Additional Context for Agent {i+1}",
                value="",
                height=100,
                help="Add any additional context or specific questions for this agent"
            )
        
        # Create agent instance if name and role are provided
        # Each agent is initialized with its specific configuration
        if agent_name and agent_role:
            # Choose agent type based on hybrid mode setting
            if use_enhanced_agents and LANGCHAIN_AVAILABLE:
                # Create enhanced agent with LangChain integration
                try:
                    enhanced_agent = create_enhanced_agent(agent_name, agent_role, selected_model, use_openai)
                    agents.append(enhanced_agent)
                    st.success(f"✅ Enhanced agent '{agent_name}' created with LangChain integration")
                except Exception as e:
                    # Fallback to standard agent if enhanced creation fails
                    st.warning(f"⚠️ Enhanced agent creation failed, using standard agent: {str(e)}")
                    agents.append(Agent(agent_name, agent_role, selected_model, use_openai))
            else:
                # Create standard agent
                agents.append(Agent(agent_name, agent_role, selected_model, use_openai))
                if not use_enhanced_agents:
                    st.info(f"📋 Standard agent '{agent_name}' created. Enable hybrid mode for LangChain features.")
    
    # ==================================================================================
    # SECTION 6: ANALYSIS CONFIGURATION
    # ==================================================================================
    
    st.subheader("Analysis Configuration")
    
    # Analysis mode selection - determines the level of automation and user interaction
    analysis_mode = st.radio(
        "Analysis Mode",
        ["Supervised", "Unsupervised", "Headless (Autonomous)"],
        help="Choose the level of agent autonomy: Supervised (manual control), Unsupervised (with self-feedback), or Headless (fully autonomous)"
    )
    
    # Additional configuration for headless mode
    # Headless mode allows fully autonomous operation with safety limits
    if analysis_mode == "Headless (Autonomous)":
        st.info("""
        🤖 **Headless Mode**: Agents will run completely autonomously with no user interaction.
        They will make their own decisions about data analysis, manipulation, and when to complete tasks.
        This demonstrates true agentic AI behavior with the workflow:
        **Analyze → Clean (if needed) → Visualize → Summarize → Repeat → End**
        """)
        
        # Safety and performance configuration for autonomous mode
        col1, col2 = st.columns(2)
        with col1:
            max_autonomous_actions = st.number_input(
                "Max Actions per Agent",
                min_value=5,
                max_value=50,
                value=15,
                help="Maximum number of actions each agent can take autonomously (safety limit)"
            )
        
        with col2:
            autonomous_delay = st.slider(
                "Action Delay (seconds)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                help="Delay between autonomous actions for better visualization and monitoring"
            )
    
    # Maximum analysis cycles configuration for supervised/unsupervised modes
    # This limits the number of complete analysis cycles per agent
    max_cycles = st.number_input(
        "Maximum Analysis Cycles",
        min_value=1,
        max_value=10,
        value=1,
        help="Number of complete analysis cycles each agent will perform (not used in Headless mode)"
    )
    
    # Set max cycles for each agent
    # This configures the stopping condition for non-autonomous modes
    for agent in agents:
        agent.set_max_cycles(max_cycles)
    
    # ==================================================================================
    # SECTION 7: ANALYSIS EXECUTION ENGINE
    # ==================================================================================
    
    # Analysis execution section - the core of the multi-agent system
    if st.button("Run Analysis"):
        st.subheader("Analysis Results")
        
        # Check if data is available in session state
        data = None
        if 'csv_data' in st.session_state:
            data = st.session_state['csv_data']
            st.info("📊 Using uploaded CSV data for analysis")
        elif 'ocr_result' in st.session_state and st.session_state['ocr_result']['success']:
            if st.session_state['ocr_result']['csv_data']:
                data = pd.DataFrame(st.session_state['ocr_result']['csv_data'])
                st.info("📄 Using OCR-processed PDF data for analysis")
            else:
                st.error("❌ No structured data available from PDF processing")
                st.stop()
        else:
            st.error("❌ No data available for analysis. Please upload a CSV file or process a PDF document first.")
            st.stop()
        
        # Create containers for organized analysis process display
        analysis_container = st.container()
        
        with analysis_container:
            # Track analysis duration for performance monitoring
            start_time = datetime.now()
            total_agents = len(agents)
            
            # ==================================================================================
            # SECTION 8: DATA ISOLATION AND PROTECTION
            # ==================================================================================
            
            # Give each agent a deep copy of the data to ensure isolation
            # This prevents agents from interfering with each other's data modifications
            for agent in agents:
                agent.set_data_copy(data)
            
            # Show data isolation confirmation to build user confidence
            st.info(f"🔒 **Data Protection**: Each agent receives an isolated copy of your data ({data.shape[0]:,} rows × {data.shape[1]} columns). The original CSV data remains unchanged.")
            
            # ==================================================================================
            # SECTION 9: AUTONOMOUS EXECUTION MODE (HEADLESS)
            # ==================================================================================
            
            # Headless mode - fully autonomous execution with real-time monitoring
            if analysis_mode == "Headless (Autonomous)":
                st.markdown("### 🤖 Autonomous Agent Execution")
                st.info("Agents are now running autonomously following the workflow: Analyze → Clean → Visualize → Summarize → Repeat. They will make their own decisions about analysis depth, data manipulation, and task completion.")
                
                # Create emergency stop functionality for safety
                kill_col, status_col = st.columns([1, 3])
                with kill_col:
                    if st.button("🛑 Kill All Agents", key="kill_agents", help="Emergency stop for all agents"):
                        st.error("🛑 All agents have been stopped!")
                        for agent in agents:
                            agent.task_complete = True
                        st.stop()
                
                # Create containers for live monitoring and logging
                log_container = st.container()  # Real-time action logging
                agent_status_container = st.container()  # Agent status grid
                
                # Track autonomous execution state
                all_agents_complete = False
                global_action_count = 0  # Safety counter to prevent infinite loops
                
                # Main autonomous execution loop
                # Continues until all agents complete or safety limit reached
                while not all_agents_complete and global_action_count < (max_autonomous_actions * total_agents):
                    all_agents_complete = True
                    
                    # Process each agent in sequence
                    for i, agent in enumerate(agents):
                        if not agent.task_complete:
                            all_agents_complete = False
                            
                            # Agent makes autonomous decision about next action
                            # This follows the workflow: Analyze → Clean → Visualize → Summarize → Repeat
                            chosen_action = agent.choose_action(f"Autonomous execution mode. Make independent decisions.")
                            
                            # Execute the chosen action based on agent's decision
                            if chosen_action == "analyze":
                                result = agent.analyze(
                                    agent.data_copy if agent.data_copy is not None else data,
                                    context="Autonomous analysis mode",
                                    custom_prompt=agent_prompt if 'agent_prompt' in locals() else None
                                )
                                action_emoji = "🔍"
                                
                            elif chosen_action == "manipulate_data":
                                result = agent.manipulate_data()
                                action_emoji = "🔧"
                                
                            elif chosen_action == "visualize":
                                result = agent.create_visualizations()
                                action_emoji = "📊"
                                
                            elif chosen_action == "summarize":
                                result = agent.summarize_findings()
                                action_emoji = "📋"
                                
                            elif chosen_action == "end":
                                result = agent.end_task()
                                action_emoji = "✅"
                            
                            else:
                                result = f"Unknown action: {chosen_action}"
                                action_emoji = "❓"
                            
                            # Log the autonomous action for real-time monitoring
                            with log_container:
                                st.markdown(f"**{action_emoji} {agent.name}** executed **{chosen_action}** - {result[:150]}{'...' if len(result) > 150 else ''}")
                            
                            # Update agent status grid for visual monitoring
                            with agent_status_container:
                                # Clear and rebuild status display to prevent stacking
                                agent_status_container.empty()
                                
                                # Create status columns for each agent
                                cols = st.columns(total_agents)
                                for idx, ag in enumerate(agents):
                                    with cols[idx]:
                                        if ag.task_complete:
                                            st.success(f"✅ {ag.name}\nCompleted")
                                        else:
                                            actions_taken = len(ag.analysis_history) + len(ag.data_manipulations)
                                            st.info(f"🔄 {ag.name}\n{actions_taken} actions taken")
                            
                            global_action_count += 1
                            
                            # Visualization delay for better user experience
                            time.sleep(autonomous_delay)
                            
                            # Break if this agent completed its task
                            if chosen_action == "end":
                                break
                    
                    # Safety check to prevent infinite loops
                    if global_action_count >= (max_autonomous_actions * total_agents):
                        st.warning("⚠️ Maximum autonomous actions reached. Stopping execution for safety.")
                        break
                
                # Final autonomous execution status
                completed_agents = sum(1 for agent in agents if agent.task_complete)
                st.success(f"✅ Autonomous execution finished! {completed_agents}/{total_agents} agents completed their tasks.")
            
            # ==================================================================================
            # SECTION 10: INTERACTIVE EXECUTION MODES (SUPERVISED/UNSUPERVISED)
            # ==================================================================================
            
            # Interactive modes (Supervised/Unsupervised) with user control
            else:
                # Process each agent with action-based workflow
                for i, agent in enumerate(agents):
                    
                    # Create containers for this agent's analysis display
                    agent_container = st.container()
                    
                    with agent_container:
                        st.markdown(f"---")
                        st.markdown(f"### Agent {i+1}: {agent.name} ({agent.role})")
                        
                        # Create containers for live updates and organized display
                        timing_container = st.container()      # Live timing and performance metrics
                        thoughts_container = st.container()    # Agent's thought process and reasoning
                        results_container = st.container()     # Analysis results and outputs
                        action_container = st.container()      # Current action and workflow step
                        
                        # Get custom prompt and context from configuration
                        agent_prompt = custom_prompt if custom_prompt != default_prompt else None
                        agent_context = analysis_context
                    
                        # Action-based workflow loop with cycle limits
                        cycle_count = 0
                        
                        # Respect max_cycles setting for interactive modes
                        # Headless mode ignores this limit for true autonomy
                        if analysis_mode == "Supervised" or analysis_mode == "Unsupervised":
                            max_actions = max_cycles * 4  # Allow multiple actions per cycle (A-C-V-S)
                        else:
                            max_actions = 20  # Default safety limit
                        
                        # Main workflow execution loop for this agent
                        while agent.can_continue_cycle() and cycle_count < max_actions:
                            cycle_count += 1
                            
                            # Check if we've exceeded cycle limit for analysis actions
                            analysis_count = len([h for h in agent.analysis_history])
                            if analysis_count >= max_cycles and analysis_mode != "Headless (Autonomous)":
                                st.info(f"🔄 {agent.name} has completed {max_cycles} analysis cycles.")
                                agent.task_complete = True
                                break
                            
                            # Display live timing and performance metrics
                            display_cycle_timing(agent, timing_container)
                            
                            # Agent chooses next action based on current workflow state
                            # This follows: Analyze → Clean → Visualize → Summarize → Repeat
                            chosen_action = agent.choose_action(agent_context)
                            
                            with action_container:
                                # Show workflow step indicators for user understanding
                                workflow_steps = {
                                    "analyze": "🔍 Step 1: Analysis",
                                    "manipulate_data": "🔧 Step 2: Data Cleaning", 
                                    "visualize": "📊 Step 3: Visualization",
                                    "summarize": "📋 Step 4: Summary",
                                    "end": "✅ Final: Task Complete"
                                }
                                
                                step_indicator = workflow_steps.get(chosen_action, f"🎯 Action: {chosen_action}")
                                st.markdown(f"#### {step_indicator}")
                                
                                # Show cycle progress for transparency
                                if chosen_action != "end":
                                    analyze_count = len([h for h in agent.analysis_history if h.get('data_manipulation') != 'Visualization creation' and not str(h.get('cycle', '')).startswith('summary')])
                                    viz_count = len(agent.visualizations)
                                    summary_count = len([h for h in agent.analysis_history if str(h.get('cycle', '')).startswith('summary')])
                                    complete_cycles = min(analyze_count, summary_count)
                                    
                                    st.caption(f"Workflow Progress: {analyze_count} analyses • {len(agent.data_manipulations)} cleanings • {viz_count} visualizations • {summary_count} summaries | Complete cycles: {complete_cycles}")
                            
                            # ==================================================================================
                            # SECTION 11: ACTION EXECUTION ENGINE
                            # ==================================================================================
                            
                            # Execute the chosen action based on workflow logic
                            if chosen_action == "analyze":
                                result = agent.analyze(
                                    agent.data_copy if agent.data_copy is not None else data,
                                    context=agent_context,
                                    custom_prompt=agent_prompt
                                )
                                action_emoji = "🔍"
                                
                            elif chosen_action == "manipulate_data":
                                result = agent.manipulate_data()
                                action_emoji = "🔧"
                                
                            elif chosen_action == "visualize":
                                result = agent.create_visualizations()
                                action_emoji = "📊"
                                
                            elif chosen_action == "summarize":
                                result = agent.summarize_findings()
                                action_emoji = "📋"
                                
                            elif chosen_action == "end":
                                result = agent.end_task()
                                action_emoji = "✅"
                            
                            else:
                                result = f"Unknown action: {chosen_action}"
                                action_emoji = "❓"
                            
                            # Update timing display after action completion
                            display_cycle_timing(agent, timing_container)
                            
                            # Display agent thoughts and reasoning process
                            display_agent_thoughts(agent, thoughts_container)
                            
                            # ==================================================================================
                            # SECTION 12: RESULTS DISPLAY AND VISUALIZATION
                            # ==================================================================================
                            
                            with results_container:
                                st.markdown(f"#### {action_emoji} {chosen_action.title()} Results")
                                
                                # Special handling for summaries to make them prominent
                                if chosen_action == "summarize":
                                    st.markdown("### 📋 Executive Summary")
                                    st.info("This is a comprehensive summary of all analysis work completed:")
                                    st.markdown(result)
                                    
                                    # Show key metrics dashboard for summary overview
                                    if agent.analysis_history:
                                        st.markdown("#### 📊 Analysis Overview")
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Analysis Cycles", len([h for h in agent.analysis_history if not str(h.get('cycle', '')).startswith('summary')]))
                                        with col2:
                                            st.metric("Visualizations", len(agent.visualizations))
                                        with col3:
                                            st.metric("Data Manipulations", len(agent.data_manipulations))
                                        with col4:
                                            if agent.data_copy is not None:
                                                st.metric("Final Data Shape", f"{agent.data_copy.shape[0]} × {agent.data_copy.shape[1]}")
                                else:
                                    # Standard result display for other actions
                                    st.write(result)
                                
                                # Show data shape changes after manipulation
                                if chosen_action == "manipulate_data" and agent.data_copy is not None:
                                    st.info(f"📊 Current data shape: {agent.data_copy.shape}")
                                
                                # Display visualizations inline when created
                                if chosen_action == "visualize" and agent.visualizations:
                                    st.markdown("### 📊 New Visualizations Created:")
                                    # Show the most recent visualizations from the latest analysis
                                    latest_analysis = None
                                    for analysis in reversed(agent.analysis_history):
                                        if 'visualizations' in analysis:
                                            latest_analysis = analysis
                                            break
                                    
                                    # Display visualizations with unique keys to prevent caching issues
                                    if latest_analysis and 'visualizations' in latest_analysis:
                                        for i, viz in enumerate(latest_analysis['visualizations']):
                                            st.markdown(f"#### {viz['title']}")
                                            st.plotly_chart(viz['figure'], use_container_width=True, key=f"inline_{agent.name}_{i}_{len(agent.analysis_history)}")
                                            st.caption(viz['description'])
                                    else:
                                        # Fallback: show last few visualizations if analysis lookup fails
                                        recent_viz = agent.visualizations[-3:] if len(agent.visualizations) >= 3 else agent.visualizations
                                        for i, viz in enumerate(recent_viz):
                                            st.markdown(f"#### {viz['title']}")
                                            st.plotly_chart(viz['figure'], use_container_width=True, key=f"inline_fallback_{agent.name}_{i}")
                                            st.caption(viz['description'])
                            
                            # ==================================================================================
                            # SECTION 13: USER INTERACTION AND FEEDBACK (SUPERVISED MODE)
                            # ==================================================================================
                            
                            # Handle feedback in supervised mode for user control
                            if analysis_mode == "Supervised" and chosen_action != "end":
                                with results_container:
                                    st.markdown("---")
                                    # User feedback input for guiding agent behavior
                                    user_feedback = st.text_area(
                                        f"Provide feedback for {agent.name} (Action {cycle_count})",
                                        placeholder="Type your feedback or leave empty to continue...",
                                        height=100,
                                        key=f"feedback_input_{i}_{cycle_count}"
                                    )
                                    
                                    # Create action buttons for user control
                                    col1, col2, col3 = st.columns(3)
                            
                                    with col1:
                                        if st.button(f"Send Feedback", key=f"feedback_{i}_{cycle_count}"):
                                            if user_feedback:
                                                response = agent.respond_to_feedback(user_feedback)
                                                st.markdown("#### Agent's Response to Feedback")
                                                st.write(response)
                            
                                    with col2:
                                        if st.button(f"Continue", key=f"next_{i}_{cycle_count}"):
                                            st.info("Continuing to next action...")
                                            break
                                    
                                    with col3:
                                        if st.button(f"Complete Task", key=f"complete_{i}_{cycle_count}"):
                                            agent.end_task()
                                            st.info("Task marked as complete")
                                            break
                                
                                # Wait for user interaction in supervised mode
                                st.stop()
                            
                            # Small delay for unsupervised mode visualization
                            elif analysis_mode == "Unsupervised":
                                time.sleep(0.5)
                            
                            # Break if task is complete
                            if chosen_action == "end" or agent.task_complete:
                                break
            
            # ==================================================================================
            # SECTION 14: COMPLETION AND TIMING SUMMARY
            # ==================================================================================
            
            # Show execution time summary
            end_time = datetime.now()
            duration = end_time - start_time
            st.markdown(f"---")
            st.info(f"Total execution time: {duration.total_seconds():.2f} seconds")
            
            # Display comprehensive analysis results from all agents
            display_analysis_results(agents, st)
        
        # ==================================================================================
        # SECTION 15: DETAILED INFORMATION DISPLAY
        # ==================================================================================
        
        # Show detailed agent information after analysis completion
        if agents:
            st.markdown("---")
            st.subheader("Detailed Agent Information")
            
            # Create tabs for different views of agent information
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Agent Status", "Visualizations", "Timing Analysis", "Data Manipulations", "Conversation History", "🔬 LangChain Stats"])
            
            with tab1:
                # Agent status overview showing goals, progress, and learning
                cols = st.columns(len(agents))
                for i, agent in enumerate(agents):
                    with cols[i]:
                        display_agent_status(agent, st.container())
            
            with tab2:
                # All visualizations created by agents in one place
                for agent in agents:
                    display_agent_visualizations(agent, st.container())
            
            with tab3:
                # Performance metrics and timing analysis
                timing_cols = st.columns(len(agents))
                for i, agent in enumerate(agents):
                    with timing_cols[i]:
                        display_cycle_timing(agent, st.container())
            
            with tab4:
                # Data manipulation history and cleaning operations
                manipulation_cols = st.columns(len(agents))
                for i, agent in enumerate(agents):
                    with manipulation_cols[i]:
                        display_data_manipulations(agent, st.container())
            
            with tab5:
                # Complete conversation history between users and agents
                for agent in agents:
                    display_conversation(agent, st.container())
            
            with tab6:
                # LangChain integration statistics and hybrid mode comparison
                if use_enhanced_agents and LANGCHAIN_AVAILABLE:
                    # Show comparison between agent types
                    display_hybrid_agent_comparison(agents, st.container())
                
                    st.markdown("---")
                    
                    # Show individual agent LangChain statistics
                    langchain_cols = st.columns(len(agents))
                    for i, agent in enumerate(agents):
                        with langchain_cols[i]:
                            display_langchain_stats(agent, st.container())
                else:
                    st.info("💡 Enable hybrid mode to see LangChain integration statistics and performance metrics.")
            
            # ==================================================================================
            # SECTION 16: POST-ANALYSIS FEEDBACK SYSTEM
            # ==================================================================================
            
            # Post-analysis feedback section for continued interaction
            st.markdown("---")
            st.subheader("Post-Analysis Feedback")
            
            # Allow users to ask questions or provide feedback after analysis completion
            user_feedback = st.text_area(
                "Provide feedback or ask questions about the completed analysis",
                placeholder="Ask questions about the results, request clarifications, or provide feedback...",
                height=100
            )
            
            if st.button("Send Post-Analysis Feedback"):
                if user_feedback:
                    # Create a container for agent responses to feedback
                    responses_container = st.container()
                    
                    with responses_container:
                        st.markdown("### Agent Responses to Feedback")
                        
                        # Get responses from each agent based on their analysis context
                        for agent in agents:
                            st.markdown(f"---")
                            st.markdown(f"#### {agent.name}'s Response")
                            
                            # Get other agents for collaborative context
                            other_agents = [a for a in agents if a != agent]
                            
                            # Get agent's response considering other agents' work
                            response = agent.respond_to_feedback(user_feedback, other_agents)
                            
                            # Display response with proper formatting
                            st.markdown(response)
                            
                            # Add a small delay between responses for better UX
                            time.sleep(0.5)

# ==================================================================================
# APPLICATION ENTRY POINT
# ==================================================================================

if __name__ == "__main__":
    # Run the main application when script is executed directly
    main() 