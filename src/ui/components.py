"""
UI Components module for the Multi-Agent Data Analysis System.

This module provides Streamlit-based UI components for displaying agent information,
analysis results, and system status. Each component is designed to present information
in a clear, organized, and visually appealing way that enhances the user experience
and provides transparency into the multi-agent analysis process.

COMPONENT ARCHITECTURE:

The UI system is built around modular, reusable components that can be combined
to create comprehensive dashboards and information displays. Each component:

- 🎨 VISUAL DESIGN: Professional, consistent styling with clear information hierarchy
- 📱 RESPONSIVE LAYOUT: Adapts to different screen sizes and container widths  
- 🔄 REAL-TIME UPDATES: Live data refresh and progressive information display
- 🎯 FOCUSED PURPOSE: Each component handles a specific aspect of information display
- 🔗 SEAMLESS INTEGRATION: Components work together to create cohesive interfaces

COMPONENT CATEGORIES:

1. 💭 AGENT INTELLIGENCE DISPLAY:
   - Agent thoughts and reasoning process visualization
   - Decision-making transparency and workflow progression
   - Goal tracking and learning point accumulation

2. 📊 ANALYSIS RESULTS PRESENTATION:
   - Comprehensive analysis results with formatting
   - Interactive visualizations with proper scaling
   - Executive summaries with business-relevant insights

3. ⚡ PERFORMANCE MONITORING:
   - Real-time timing information and cycle progress
   - Token usage tracking and speed calculations
   - System resource utilization and model requirements

4. 🔧 TECHNICAL INFORMATION:
   - Data manipulation history and cleaning operations
   - Model recommendations based on system capabilities
   - Conversation history with chronological organization

KEY DESIGN PRINCIPLES:

🎯 CLARITY: Information is presented in digestible, well-organized sections
📈 PROGRESSION: Clear indication of workflow progress and completion status  
🔍 TRANSPARENCY: Full visibility into agent reasoning and decision-making
⚡ PERFORMANCE: Efficient rendering with minimal computational overhead
🎨 CONSISTENCY: Uniform styling and layout patterns across all components
🔄 REAL-TIME: Live updates that reflect current system state accurately

INTEGRATION PATTERNS:

The components integrate with the agent system through standardized interfaces:
- Agent objects provide data through well-defined attributes
- Streamlit containers enable flexible layout and organization
- Real-time updates through progressive rendering and state management
- Error handling ensures graceful degradation when data is unavailable

This modular approach enables easy customization, testing, and maintenance
while providing a rich, interactive experience for users monitoring and
controlling the multi-agent data analysis process.
"""

import streamlit as st
import time
from datetime import datetime
from typing import List, Dict
from src.models.agent import Agent
import pandas as pd
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.config.agent_roles import get_enhanced_role_prompts, get_analysis_task_templates
    from src.utils.advanced_visualizations import AdvancedVisualizationEngine, recommend_visualizations
    from src.utils.web_tools import create_web_tool
except ImportError as e:
    print(f"Note: Some advanced features not available: {e}")
    get_enhanced_role_prompts = None
    get_analysis_task_templates = None
    AdvancedVisualizationEngine = None
    recommend_visualizations = None
    create_web_tool = None

def display_agent_thoughts(agent: Agent, container):
    """
    Display the agent's thought process in a visually appealing way.
    
    This component shows the agent's thoughts in a sequential manner with
    a slight delay between each thought to create an animated effect.
    
    Args:
        agent (Agent): Agent whose thoughts to display
        container (streamlit.container): Streamlit container to display thoughts in
    """
    with container:
        st.markdown(f"### {agent.name}'s Thought Process")
        for thought in agent.thoughts:
            st.markdown(f"💭 {thought}")
            time.sleep(0.5)  # Animate the thoughts appearing

def display_conversation(agent: Agent, container):
    """
    Display the agent's conversation history in a visually appealing way.
    
    This component shows the agent's conversation history with timestamps
    and role information, formatted for easy reading.
    
    Args:
        agent (Agent): Agent whose conversation to display
        container (streamlit.container): Streamlit container to display conversation in
    """
    with container:
        st.markdown(f"### {agent.name}'s Conversation History")
        for message in agent.conversation_history:
            if message["role"] == "assistant":
                st.markdown(f"**{agent.name} ({agent.role})** - {message['timestamp']}")
                st.markdown(message["content"])
                st.markdown("---")

def display_agent_status(agent: Agent, container):
    """
    Display the agent's current status, goals, and learning points.
    
    This component provides a comprehensive view of the agent's current state,
    including its goals, completed goals, learning points, and improvement
    suggestions.
    
    Args:
        agent (Agent): Agent whose status to display
        container (streamlit.container): Streamlit container to display status in
    """
    with container:
        st.markdown(f"### {agent.name}'s Status")
        
        # Display current goals
        st.markdown("#### Current Goals")
        st.markdown(f"**Primary Goal:** {agent.goals['primary_goal']}")
        st.markdown(f"**Current Focus:** {agent.goals['current_focus'] if agent.goals['current_focus'] else 'None'}")
        
        # Display completed goals
        if agent.goals['completed_goals']:
            st.markdown("**Completed Goals:**")
            for goal in agent.goals['completed_goals']:
                st.markdown(f"- ✅ {goal}")
        
        # Display learning points
        if agent.learning_points:
            st.markdown("#### Learning Points")
            for point in agent.learning_points[-5:]:  # Show last 5 learning points
                st.markdown(f"- 💡 {point}")
        
        # Display improvement suggestions
        st.markdown("#### Improvement Suggestions")
        suggestions = agent._generate_improvement_suggestions()
        st.markdown(suggestions)

def display_cycle_timing(agent: Agent, container):
    """
    Display the timing information for each analysis cycle with live updates.
    
    This component shows detailed timing information for each analysis cycle,
    including a live timer for the current cycle, progress bar, and token usage.
    
    Args:
        agent (Agent): Agent whose timing to display
        container (streamlit.container): Streamlit container to display timing in
    """
    # Clear the container content to prevent stacking
    container.empty()
    
    with container:
        st.markdown(f"### ⏱️ {agent.name} - Cycle {agent.cycle_count}/{agent.max_cycles}")
    
        # Current cycle live timer (only if running)
        if agent.current_cycle_start and agent.cycle_count <= agent.max_cycles:
            elapsed_time = (datetime.now() - agent.current_cycle_start).total_seconds()
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            
            st.markdown(f"🟢 **Running** - Live Timer: {time_str}")
        
            # Token usage if available
            if hasattr(agent, 'current_tokens_used') and agent.current_tokens_used > 0:
                tokens_per_second = agent.current_tokens_used / elapsed_time if elapsed_time > 0 else 0
                st.markdown(f"🔤 Tokens: {agent.current_tokens_used:,} | Speed: {tokens_per_second:.1f} t/s")
        
        # Cycle history table
        if agent.cycle_times:
            st.markdown("**📈 Cycle History**")
            
            timing_data = []
            for t in agent.cycle_times:
                tokens_used = t.get('tokens_used', 0)
                tokens_per_sec = t.get('tokens_per_second', 0)
                
                # Convert cycle to string to avoid serialization issues
                cycle_str = str(t["cycle"]) if t["cycle"] is not None else "N/A"
                
                timing_data.append({
                    "Cycle": cycle_str,
                    "Duration": f"{t['time_seconds']:.1f}s",
                    "Tokens": f"{tokens_used:,}" if isinstance(tokens_used, (int, float)) and tokens_used > 0 else 'N/A',
                    "Speed": f"{tokens_per_sec:.1f} t/s" if isinstance(tokens_per_sec, (int, float)) and tokens_per_sec > 0 else 'N/A'
                })
            
            df = pd.DataFrame(timing_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        elif agent.cycle_count == 0:
            st.info("🔄 Ready to start analysis")
        else:
            st.info("⏳ Analysis starting...")

def display_data_manipulations(agent: Agent, container):
    """
    Display the data manipulation suggestions for each cycle.
    
    This component shows the agent's data manipulation suggestions in an
    expandable format, organized by cycle and timestamp.
    
    Args:
        agent (Agent): Agent whose manipulations to display
        container (streamlit.container): Streamlit container to display manipulations in
    """
    with container:
        st.markdown(f"### {agent.name}'s Data Manipulations")
        
        if agent.data_manipulations:
            for manipulation in agent.data_manipulations:
                with st.expander(f"Cycle {manipulation['cycle']} - {manipulation['timestamp']}"):
                    st.markdown("#### Suggested Data Manipulations")
                    st.markdown(manipulation['suggestion'])
        else:
            st.markdown("No data manipulation suggestions available yet.")

def display_analysis_results(agents: List[Agent], container):
    """
    Display the analysis results from all agents.
    
    This component shows the analysis results from each agent in an organized
    format, with expandable sections for each cycle and download options.
    
    Args:
        agents (List[Agent]): List of agents to display results from
        container (streamlit.container): Streamlit container to display results in
    """
    container.markdown("### Analysis Results")
    
    for agent in agents:
        container.markdown(f"---")
        container.markdown(f"#### Agent: {agent.name} ({agent.role})")
        
        # Display analysis history
        for analysis in agent.analysis_history:
            with container.expander(f"Cycle {analysis['cycle']} - {analysis['timestamp']} (Time: {analysis['cycle_time']}s)"):
                container.markdown("#### Analysis Results")
                container.markdown(analysis['results'])
                
                container.markdown("#### Data Manipulation Suggestions")
                container.markdown(analysis['data_manipulation'])
        
        # Add download button for this agent's results
        results_text = "\n\n".join([
            f"Cycle {cycle['cycle']} ({cycle['cycle_time']}s):\n"
            f"Data Manipulation:\n{cycle['data_manipulation']}\n\n"
            f"Analysis Results:\n{cycle['results']}"
            for cycle in agent.analysis_history
        ])
        
        container.download_button(
            label=f"Download {agent.name}'s Analysis Results",
            data=results_text,
            file_name=f"{agent.name}_analysis_results.txt",
            mime="text/plain"
        )

def display_model_recommendations(recommendations: Dict, use_openai: bool, container) -> None:
    """
    Display model recommendations for the agent.
    
    This component shows recommended models based on the agent's role and
    goals, differentiating between OpenAI and Ollama models.
    
    Args:
        recommendations: Dictionary containing model recommendations
        use_openai: Whether to use OpenAI models
        container: Streamlit container to display recommendations in
    """
    container.markdown("### Recommended Models")
    
    if use_openai:
        container.markdown("#### OpenAI Models")
        for model in recommendations["openai_models"]:
            container.markdown(f"- **{model['name']}**: {model['reason']}")
    else:
        container.markdown("#### Ollama Models")
        for model in recommendations["ollama_models"]:
            container.markdown(f"- **{model['name']}**: {model['reason']}")

def display_system_resources(resources: dict, container):
    """
    Display system resources information.
    
    This component shows information about available system resources,
    including memory and CPU cores.
    
    Args:
        resources (dict): Dictionary containing system resources
        container (streamlit.container): Streamlit container to display resources in
    """
    with container:
        st.markdown("### System Resources")
        st.markdown(f"- Available Memory: {resources['memory_gb']} GB")
        st.markdown(f"- CPU Cores: {resources['cpu_count']}")

def display_model_requirements(model_requirements: dict, resources: dict, container):
    """
    Display model requirements and recommendations.
    
    This component shows the requirements for each model and indicates
    whether the system meets those requirements.
    
    Args:
        model_requirements (dict): Dictionary of model requirements
        resources (dict): Dictionary of system resources
        container (streamlit.container): Streamlit container to display requirements in
    """
    with container:
        st.markdown("### Model Requirements")
        
        for model, reqs in model_requirements.items():
            if resources['memory_gb'] >= reqs['min_memory_gb']:
                st.markdown(f"✅ {model} ({reqs['size_gb']:.2f} GB)")
                st.markdown(f"   {reqs['description']}")
                st.markdown(f"   Min RAM: {reqs['min_memory_gb']}GB | Rec RAM: {reqs['recommended_memory_gb']}GB")
            else:
                st.markdown(f"⚠️ {model} ({reqs['size_gb']:.2f} GB)")
                st.markdown(f"   {reqs['description']}")
                st.markdown(f"   Requires {reqs['min_memory_gb']}GB RAM")
            st.markdown("---") 

def display_agent_visualizations(agent, container):
    """
    Display visualizations created by an agent.
    
    Args:
        agent: The agent whose visualizations to display
        container: Streamlit container to display in
    """
    with container:
        if agent.visualizations:
            st.markdown(f"### 📊 Visualizations by {agent.name}")
            
            for i, viz in enumerate(agent.visualizations):
                st.markdown(f"#### {viz['title']}")
                st.plotly_chart(viz['figure'], use_container_width=True, key=f"{agent.name}_viz_{i}")
                st.caption(viz['description'])
                st.markdown("---")
        else:
            st.info(f"No visualizations created by {agent.name} yet.")

def display_langchain_stats(agent, container):
    """
    Display LangChain integration statistics and performance metrics for enhanced agents.
    
    This component shows the performance and usage statistics of LangChain integration,
    including success rates, response times, token usage, and tool availability.
    
    Args:
        agent: Enhanced agent with LangChain integration
        container: Streamlit container to display stats in
    """
    with container:
        # Check if this is an enhanced agent with LangChain capabilities
        if hasattr(agent, 'get_langchain_stats'):
            stats = agent.get_langchain_stats()
            
            st.markdown(f"### 🔬 LangChain Integration Stats - {agent.name}")
            
            # Integration status
            if stats.get('langchain_enabled', False):
                st.success("✅ LangChain integration active")
            else:
                st.error("❌ LangChain integration disabled")
            
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Enhanced Responses", 
                    stats.get('total_enhanced_responses', 0),
                    help="Total number of responses generated using LangChain"
                )
                
            with col2:
                success_rate = stats.get('success_rate', 0)
                st.metric(
                    "Success Rate", 
                    f"{success_rate:.1f}%",
                    help="Percentage of successful LangChain responses"
                )
                
            with col3:
                avg_time = stats.get('avg_response_time', 0)
                st.metric(
                    "Avg Response Time", 
                    f"{avg_time:.2f}s",
                    help="Average time for LangChain-enhanced responses"
                )
            
            # Additional details in expandable section
            with st.expander("📊 Detailed LangChain Metrics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Usage Statistics:**")
                    st.markdown(f"- Successful responses: {stats.get('successful_responses', 0)}")
                    st.markdown(f"- Total tokens used: {stats.get('total_tokens_used', 0):,}")
                    st.markdown(f"- Tools available: {stats.get('tools_available', 0)}")
                
                with col2:
                    st.markdown("**Performance Indicators:**")
                    if success_rate >= 90:
                        st.markdown("🟢 Excellent performance")
                    elif success_rate >= 70:
                        st.markdown("🟡 Good performance")
                    else:
                        st.markdown("🔴 Performance issues detected")
                    
                    if avg_time <= 2.0:
                        st.markdown("⚡ Fast response times")
                    elif avg_time <= 5.0:
                        st.markdown("⏱️ Moderate response times")
                    else:
                        st.markdown("🐌 Slow response times")
            
            # Toggle button for LangChain integration
            if hasattr(agent, 'toggle_langchain'):
                current_status = "Enabled" if stats.get('langchain_enabled', False) else "Disabled"
                new_status = "Disable" if stats.get('langchain_enabled', False) else "Enable"
                
                if st.button(f"{new_status} LangChain", key=f"toggle_{agent.name}"):
                    new_state = agent.toggle_langchain()
                    status_text = "enabled" if new_state else "disabled"
                    st.success(f"LangChain integration {status_text} for {agent.name}")
                    st.experimental_rerun()
        else:
            st.info(f"{agent.name} is using standard agent implementation. Enable hybrid mode for LangChain statistics.")

def display_hybrid_agent_comparison(agents, container):
    """
    Display a comparison between standard and enhanced agents.
    
    Args:
        agents: List of agents (mix of standard and enhanced)
        container: Streamlit container to display comparison in
    """
    with container:
        st.markdown("### 🔄 Agent Type Comparison")
        
        # Separate agents by type
        standard_agents = []
        enhanced_agents = []
        
        for agent in agents:
            if hasattr(agent, 'get_langchain_stats'):
                enhanced_agents.append(agent)
            else:
                standard_agents.append(agent)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Standard Agents")
            if standard_agents:
                for agent in standard_agents:
                    st.markdown(f"- **{agent.name}** ({agent.role})")
                    st.caption(f"Analysis cycles: {len(agent.analysis_history)}")
            else:
                st.info("No standard agents")
        
        with col2:
            st.markdown("#### 🚀 Enhanced Agents")
            if enhanced_agents:
                for agent in enhanced_agents:
                    stats = agent.get_langchain_stats()
                    status = "🟢" if stats.get('langchain_enabled', False) else "🔴"
                    st.markdown(f"- **{agent.name}** ({agent.role}) {status}")
                    st.caption(f"Enhanced responses: {stats.get('total_enhanced_responses', 0)}")
            else:
                st.info("No enhanced agents")
        
        # Summary statistics
        if enhanced_agents:
            st.markdown("#### 📊 Hybrid Mode Summary")
            total_enhanced_responses = sum(
                agent.get_langchain_stats().get('total_enhanced_responses', 0) 
                for agent in enhanced_agents
            )
            avg_success_rate = sum(
                agent.get_langchain_stats().get('success_rate', 0) 
                for agent in enhanced_agents
            ) / len(enhanced_agents)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Enhanced Agents", len(enhanced_agents))
            with col2:
                st.metric("Total Enhanced Responses", total_enhanced_responses)
            with col3:
                st.metric("Avg Success Rate", f"{avg_success_rate:.1f}%")

def display_task_selection_interface():
    """
    Display an interface for users to select analysis tasks and goals.
    
    Returns:
        dict: Selected task configuration
    """
    st.markdown("### 🎯 Analysis Task Selection")
    
    # Get available task templates
    if get_analysis_task_templates:
        task_templates = get_analysis_task_templates()
        
        # Task template selection
        task_names = list(task_templates.keys())
        selected_task = st.selectbox(
            "Select Analysis Type:",
            task_names,
            help="Choose a predefined analysis template or create a custom analysis"
        )
        
        if selected_task:
            task_config = task_templates[selected_task]
            
            # Display task information
            st.markdown(f"**{task_config['name']}**")
            st.markdown(task_config['description'])
            
            # Display recommended agents
            st.markdown("**Recommended Agent Types:**")
            for agent_type in task_config['recommended_agents']:
                st.markdown(f"- {agent_type}")
            
            # Allow customization of tasks
            st.markdown("**Analysis Tasks:**")
            custom_tasks = []
            
            for i, task in enumerate(task_config['tasks']):
                include_task = st.checkbox(f"{task}", value=True, key=f"task_{i}")
                if include_task:
                    custom_tasks.append(task)
            
            # Option to add custom tasks
            st.markdown("**Additional Custom Tasks:**")
            custom_task = st.text_area(
                "Add custom analysis tasks (one per line):",
                help="Enter additional specific tasks for the analysis"
            )
            
            if custom_task:
                custom_tasks.extend([task.strip() for task in custom_task.split('\n') if task.strip()])
            
            return {
                'template_name': selected_task,
                'name': task_config['name'],
                'description': task_config['description'],
                'tasks': custom_tasks,
                'recommended_agents': task_config['recommended_agents']
            }
    
    # Fallback for manual task creation
    else:
        st.markdown("**Manual Task Configuration:**")
        
        task_name = st.text_input("Task Name:", value="Custom Analysis")
        task_description = st.text_area("Task Description:", value="Custom data analysis task")
        
        custom_tasks_text = st.text_area(
            "Analysis Tasks (one per line):",
            value="Exploratory data analysis\nIdentify key patterns\nProvide actionable insights"
        )
        
        custom_tasks = [task.strip() for task in custom_tasks_text.split('\n') if task.strip()]
        
        return {
            'template_name': 'custom',
            'name': task_name,
            'description': task_description,
            'tasks': custom_tasks,
            'recommended_agents': ['Data Scientist', 'Business Analyst']
        }

def display_enhanced_agent_selection():
    """
    Display enhanced agent selection with role descriptions and capabilities.
    
    Returns:
        dict: Selected agent configuration
    """
    st.markdown("### 🤖 Enhanced Agent Selection")
    
    if get_enhanced_role_prompts:
        role_prompts = get_enhanced_role_prompts()
        
        # Agent role selection
        role_names = list(role_prompts.keys())
        selected_role = st.selectbox(
            "Select Agent Role:",
            role_names,
            help="Choose an agent with specialized expertise for your analysis"
        )
        
        if selected_role:
            role_config = role_prompts[selected_role]
            
            # Display role information in an expandable section
            with st.expander("View Role Details & Capabilities", expanded=False):
                st.markdown("**Role Description:**")
                # Show first few lines of the prompt
                prompt_preview = role_config['prompt'][:500] + "..." if len(role_config['prompt']) > 500 else role_config['prompt']
                st.markdown(prompt_preview)
                
                # Display tools and capabilities
                if 'tools' in role_config:
                    st.markdown("**Available Tools:**")
                    for tool in role_config['tools']:
                        st.markdown(f"- {tool}")
                
                if 'capabilities' in role_config:
                    st.markdown("**Key Capabilities:**")
                    for capability in role_config['capabilities']:
                        st.markdown(f"- {capability}")
            
            return {
                'role': selected_role,
                'prompt': role_config['prompt'],
                'tools': role_config.get('tools', []),
                'capabilities': role_config.get('capabilities', []),
                'visualization_focus': role_config.get('visualization_focus', [])
            }
    
    # Fallback to basic role selection
    else:
        basic_roles = ['Business Analyst', 'Data Scientist', 'Statistician', 'Accountant', 'Sports Coach', 'Domain Expert', 'Custom']
        selected_role = st.selectbox("Select Agent Role:", basic_roles)
        
        return {
            'role': selected_role,
            'prompt': f"You are an expert {selected_role}. Analyze the data and provide insights.",
            'tools': [],
            'capabilities': [],
            'visualization_focus': []
        }

def display_enhanced_visualizations(agent, data, container):
    """
    Display enhanced visualizations with advanced chart types.
    
    Args:
        agent: Agent object
        data: DataFrame with analysis data
        container: Streamlit container
    """
    with container:
        st.markdown("### 📊 Enhanced Visualizations")
        
        if data is None or data.empty:
            st.info("No data available for visualization")
            return
        
        # Initialize advanced visualization engine
        if AdvancedVisualizationEngine:
            viz_engine = AdvancedVisualizationEngine()
            
            # Get visualization recommendations
            if recommend_visualizations:
                recommendations = recommend_visualizations(data, agent.role)
                
                if recommendations:
                    st.markdown("**Recommended Visualizations:**")
                    for rec in recommendations[:3]:  # Show top 3 recommendations
                        st.markdown(f"- **{rec['type'].replace('_', ' ').title()}** ({rec['priority']} priority): {rec['description']}")
            
            # Visualization options
            viz_types = [
                "Advanced Correlation Heatmap",
                "3D Scatter Plot",
                "Distribution Comparison",
                "Time Series Decomposition",
                "Financial Dashboard",
                "Sports Performance Chart",
                "Scatter Plot Matrix"
            ]
            
            selected_viz = st.selectbox("Select Visualization Type:", viz_types)
            
            try:
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
                date_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
                
                if selected_viz == "Advanced Correlation Heatmap" and len(numeric_cols) >= 2:
                    correlation_method = st.radio("Correlation Method:", ["pearson", "spearman", "kendall"])
                    fig = viz_engine.create_correlation_heatmap_advanced(data, correlation_method)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif selected_viz == "3D Scatter Plot" and len(numeric_cols) >= 3:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_col = st.selectbox("X-axis:", numeric_cols, key="x_3d")
                    with col2:
                        y_col = st.selectbox("Y-axis:", numeric_cols, key="y_3d", index=1 if len(numeric_cols) > 1 else 0)
                    with col3:
                        z_col = st.selectbox("Z-axis:", numeric_cols, key="z_3d", index=2 if len(numeric_cols) > 2 else 0)
                    
                    color_col = st.selectbox("Color by (optional):", ["None"] + numeric_cols + categorical_cols, key="color_3d")
                    color_col = None if color_col == "None" else color_col
                    
                    fig = viz_engine.create_3d_scatter(data, x_col, y_col, z_col, color_col)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif selected_viz == "Distribution Comparison" and len(numeric_cols) >= 2:
                    selected_cols = st.multiselect("Select columns to compare:", numeric_cols, default=numeric_cols[:3])
                    chart_type = st.radio("Chart type:", ["violin", "box", "histogram"])
                    
                    if selected_cols:
                        fig = viz_engine.create_distribution_comparison(data, selected_cols, chart_type)
                        st.plotly_chart(fig, use_container_width=True)
                
                elif selected_viz == "Time Series Decomposition" and date_cols and numeric_cols:
                    date_col = st.selectbox("Date column:", date_cols)
                    value_col = st.selectbox("Value column:", numeric_cols)
                    
                    fig = viz_engine.create_time_series_decomposition(data, date_col, value_col)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif selected_viz == "Financial Dashboard" and agent.role == "Accountant":
                    # Calculate basic financial metrics for demonstration
                    financial_metrics = {}
                    if 'revenue' in data.columns:
                        financial_metrics['revenue_growth'] = data['revenue'].pct_change().mean() * 100
                    if 'profit' in data.columns and 'revenue' in data.columns:
                        financial_metrics['profit_margin'] = (data['profit'] / data['revenue']).mean() * 100
                    financial_metrics['roi'] = 15.5  # Example ROI
                    
                    fig = viz_engine.create_financial_dashboard(data, financial_metrics)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif selected_viz == "Sports Performance Chart" and agent.role == "Sports Coach":
                    if 'player' in data.columns or 'name' in data.columns:
                        player_col = 'player' if 'player' in data.columns else 'name'
                        metric_cols = st.multiselect("Performance metrics:", numeric_cols, default=numeric_cols[:3])
                        
                        if metric_cols:
                            fig = viz_engine.create_sports_performance_chart(data, player_col, metric_cols)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Sports performance chart requires a 'player' or 'name' column")
                
                elif selected_viz == "Scatter Plot Matrix" and len(numeric_cols) >= 3:
                    selected_dims = st.multiselect("Select dimensions:", numeric_cols, default=numeric_cols[:4])
                    color_col = st.selectbox("Color by (optional):", ["None"] + categorical_cols, key="color_matrix")
                    color_col = None if color_col == "None" else color_col
                    
                    if len(selected_dims) >= 3:
                        fig = viz_engine.create_advanced_scatter_matrix(data, selected_dims, color_col)
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.info(f"Selected visualization requires specific data types or agent role. Please check your data or select a different visualization.")
            
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
        
        else:
            st.info("Advanced visualizations require additional dependencies. Using basic charts.")
            # Fallback to basic visualizations
            if not data.empty:
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_cols) >= 2:
                    import plotly.express as px
                    fig = px.scatter_matrix(data[numeric_cols[:4]])
                    st.plotly_chart(fig, use_container_width=True)

def display_headless_mode_status(agents: List[Agent], container):
    """
    Display enhanced status information for headless mode with real-time updates.
    
    Args:
        agents: List of agent objects
        container: Streamlit container
    """
    with container:
        st.markdown("### 🤖 Agent Activity Monitor")
        
        # Overall progress
        total_agents = len(agents)
        active_agents = sum(1 for agent in agents if hasattr(agent, 'current_cycle_start') and agent.current_cycle_start)
        completed_agents = sum(1 for agent in agents if hasattr(agent, 'cycle_count') and hasattr(agent, 'max_cycles') and agent.cycle_count >= agent.max_cycles)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Agents", total_agents)
        with col2:
            st.metric("Active", active_agents)
        with col3:
            st.metric("Completed", completed_agents)
        
        # Progress bar
        overall_progress = completed_agents / total_agents if total_agents > 0 else 0
        st.progress(overall_progress)
        
        # Individual agent status
        st.markdown("**Individual Agent Status:**")
        
        for agent in agents:
            with st.expander(f"{agent.name} ({agent.role})", expanded=active_agents <= 2):
                # Current activity
                if hasattr(agent, 'current_cycle_start') and agent.current_cycle_start:
                    elapsed = (datetime.now() - agent.current_cycle_start).total_seconds()
                    st.markdown(f"🟢 **Status:** Active (Running for {elapsed:.1f}s)")
                elif hasattr(agent, 'cycle_count') and hasattr(agent, 'max_cycles') and agent.cycle_count >= agent.max_cycles:
                    st.markdown("✅ **Status:** Completed")
                else:
                    st.markdown("⏸️ **Status:** Idle")
                
                # Progress information
                if hasattr(agent, 'cycle_count') and hasattr(agent, 'max_cycles'):
                    progress = agent.cycle_count / agent.max_cycles if agent.max_cycles > 0 else 0
                    st.progress(progress)
                    st.markdown(f"Cycle {agent.cycle_count}/{agent.max_cycles}")
                
                # Recent thoughts/actions
                if hasattr(agent, 'thoughts') and agent.thoughts:
                    st.markdown("**Recent Thoughts:**")
                    for thought in agent.thoughts[-3:]:  # Show last 3 thoughts
                        st.markdown(f"💭 {thought}")
                
                # Performance metrics
                if hasattr(agent, 'cycle_times') and agent.cycle_times:
                    avg_time = sum(t.get('time_seconds', 0) for t in agent.cycle_times) / len(agent.cycle_times)
                    st.markdown(f"**Performance:** Avg cycle time: {avg_time:.1f}s")

def display_web_search_results(search_results: Dict, container):
    """
    Display web search results in an organized format.
    
    Args:
        search_results: Dictionary containing search results
        container: Streamlit container
    """
    with container:
        st.markdown("### 🌐 Web Search Results")
        
        if 'error' in search_results:
            st.error(f"Search error: {search_results['error']}")
            return
        
        # Search query and metadata
        if 'query' in search_results:
            st.markdown(f"**Search Query:** {search_results['query']}")
        
        if 'timestamp' in search_results:
            st.markdown(f"**Search Time:** {search_results['timestamp']}")
        
        # News sentiment analysis
        if 'overall_sentiment' in search_results:
            sentiment_color = {'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'}
            sentiment_icon = sentiment_color.get(search_results['overall_sentiment'], '🟡')
            st.markdown(f"**Overall Sentiment:** {sentiment_icon} {search_results['overall_sentiment'].title()}")
        
        # Articles or search results
        if 'articles' in search_results:
            st.markdown("**Articles Found:**")
            for article in search_results['articles'][:5]:  # Show top 5 articles
                with st.expander(article['title']):
                    st.markdown(article['snippet'])
                    if 'sentiment' in article:
                        sentiment_icon = sentiment_color.get(article['sentiment'], '🟡')
                        st.markdown(f"Sentiment: {sentiment_icon} {article['sentiment']}")
                    if 'link' in article:
                        st.markdown(f"[Read more]({article['link']})")
        
        elif 'search_results' in search_results:
            st.markdown("**Search Results:**")
            for result in search_results['search_results'][:5]:
                with st.expander(result['title']):
                    st.markdown(result['snippet'])
                    if 'link' in result:
                        st.markdown(f"[Read more]({result['link']})")
        
        # Additional information
        if 'potential_competitors' in search_results:
            st.markdown("**Potential Competitors:**")
            for competitor in search_results['potential_competitors']:
                st.markdown(f"- {competitor}")
        
        if 'key_insights' in search_results:
            st.markdown("**Key Insights:**")
            for insight in search_results['key_insights']:
                st.markdown(f"- {insight}") 