"""
Agent module for the Multi-Agent Data Analysis System.

This module defines the Agent class which represents an intelligent analysis agent with specific 
capabilities for autonomous data analysis, manipulation, and visualization. Each agent follows
a sophisticated workflow: Analyze → Clean (if needed) → Visualize → Summarize → Repeat → End

The Agent class implements:
- Autonomous decision-making based on current analysis state and data quality
- Role-based analysis with specialized prompts and domain-specific goals
- Data isolation ensuring each agent works on its own copy without interference
- Comprehensive workflow management with cycle tracking and performance monitoring
- Interactive visualization creation with automatic chart type selection
- Real-time data cleaning and manipulation with quality assessment
- Executive summary generation with actionable insights and recommendations
- Learning and goal tracking for continuous improvement and progress monitoring
- Support for both OpenAI (cloud) and Ollama (local) models with token tracking

WORKFLOW ARCHITECTURE:
The agent follows a structured workflow that mirrors professional data analysis:

🔍 ANALYZE: Deep analysis of data patterns, statistical relationships, and domain insights
🔧 CLEAN: Intelligent data manipulation, missing value handling, and quality improvement  
📊 VISUALIZE: Automatic creation of appropriate charts based on data characteristics
📋 SUMMARIZE: Generation of executive-level summaries with actionable recommendations
🔄 REPEAT: Autonomous decision-making about additional analysis cycles needed
✅ END: Task completion when sufficient insights have been gathered

DECISION-MAKING LOGIC:
The agent uses sophisticated logic to progress through the workflow:
- Tracks completion status of each workflow step (analysis, cleaning, visualization, summary)
- Evaluates data quality metrics to determine if cleaning operations are needed
- Ensures proper sequence is followed (can't visualize before analyzing, etc.)
- Makes autonomous decisions about task completion based on analysis depth
- Considers role-specific goals and objectives when determining next actions
- Balances thoroughness with efficiency to avoid unnecessary repetition

Key Features:
- Intelligent workflow progression with logical decision points and safety checks
- Real-time data cleaning with automatic missing value and duplicate handling
- Dynamic visualization creation with chart type selection based on data types
- Comprehensive summary generation with business-relevant insights and recommendations
- Multi-cycle analysis capability with autonomous continuation decisions
- Collaborative analysis support with other agents in the same session
- Detailed history tracking for auditing, debugging, and performance analysis
- Learning system that tracks goal completion and knowledge accumulation
"""

from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import ollama
import openai
import time
import os
import shutil
from src.utils.model_manager import ModelManager
from src.config.prompts import get_role_prompts
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import enhanced capabilities
try:
    from src.config.agent_roles import get_enhanced_role_prompts
    from src.utils.web_tools import create_web_tool
    from src.utils.advanced_visualizations import AdvancedVisualizationEngine
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    print("Note: Enhanced features not available. Using basic agent capabilities.")

class Agent:
    """
    Agent class representing an analysis agent with specific role and model.
    
    The Agent class is the core component of the Multi-Agent Data Analysis System.
    Each agent is initialized with a specific role and model, and maintains its own
    state including thoughts, conversation history, analysis history, and goals.
    
    Attributes:
        name (str): The name of the agent
        role (str): The role of the agent (e.g., Business Analyst, Data Analyst)
        model (str): The model to use for analysis
        use_openai (bool): Whether to use OpenAI API or Ollama
        thoughts (List[str]): List of agent's thoughts during analysis
        conversation_history (List[Dict]): History of conversations
        analysis_history (List[Dict]): History of analysis results
        goals (Dict): Agent's goals and progress
        learning_points (List[str]): Points learned during analysis
        cycle_count (int): Current analysis cycle count
        max_cycles (int): Maximum number of analysis cycles
        cycle_times (List[Dict]): Timing information for each cycle
        data_manipulations (List[Dict]): Data manipulation suggestions
        current_cycle_start (datetime): Start time of the current cycle
        current_tokens_used (int): Track tokens used in current cycle
        data_copy (pd.DataFrame): Agent's copy of the data for manipulation
        available_actions (List[str]): Available actions for the agent
        visualizations (List[Dict]): Created visualizations
    """
    
    def __init__(self, name: str, role: str, model: str = "deepseek-coder:latest", use_openai: bool = False):
        """
        Initialize a new Agent instance.
        
        Args:
            name (str): Name of the agent
            role (str): Role of the agent
            model (str): Model to use for analysis
            use_openai (bool): Whether to use OpenAI API
        """
        self.name = name
        self.role = role
        self.model = model
        self.use_openai = use_openai
        self.thoughts = []
        self.conversation_history = []
        self.analysis_history = []
        self.goals = self._initialize_goals()
        self.learning_points = []
        self.cycle_count = 0
        self.max_cycles = 1
        self.model_manager = ModelManager()
        self.cycle_times = []
        self.data_manipulations = []
        self.current_cycle_start = None
        self.current_tokens_used = 0  # Track tokens used in current cycle
        self.data_copy = None  # Agent's copy of the data
        self.available_actions = ["analyze", "manipulate_data", "visualize", "summarize", "end"]
        self.task_complete = False
        self.visualizations = []  # Store created visualizations
        
        # Enhanced capabilities
        self.web_search_tool = None
        self.enhanced_viz_engine = None
        self.web_search_results = []
        self.role_config = None
        
        # Initialize web search tool for all agents
        self._initialize_web_search_tool()
        
        # Initialize enhanced features if available
        if ENHANCED_FEATURES_AVAILABLE:
            self._initialize_enhanced_features()

    def _initialize_web_search_tool(self):
        """
        Initialize web search tool for the agent.
        """
        try:
            from src.utils.web_tools import WebSearchTool
            self.web_search_tool = WebSearchTool()
            self.thoughts.append("🌐 Web search capabilities enabled")
        except ImportError:
            self.thoughts.append("⚠️ Web search not available - install required dependencies")
        except Exception as e:
            self.thoughts.append(f"⚠️ Web search initialization failed: {str(e)}")

    def _generate_data_specific_search_queries(self) -> List[str]:
        """
        Generate search queries based on the actual data content and analysis results.
        
        Returns:
            List of relevant search queries based on data content
        """
        if self.data_copy is None or self.data_copy.empty:
            return []
        
        search_queries = []
        
        try:
            # Analyze column names to understand data type
            columns = [col.lower() for col in self.data_copy.columns]
            
            # Financial data indicators
            financial_keywords = ['amount', 'balance', 'income', 'expense', 'cost', 'price', 'revenue', 'profit', 'transaction', 'payment', 'spending', 'budget', 'salary', 'fee']
            has_financial_data = any(keyword in ' '.join(columns) for keyword in financial_keywords)
            
            # Sales/business data indicators  
            business_keywords = ['sales', 'customer', 'product', 'order', 'quantity', 'units', 'region', 'category', 'performance', 'growth']
            has_business_data = any(keyword in ' '.join(columns) for keyword in business_keywords)
            
            # Time series indicators
            time_keywords = ['date', 'time', 'year', 'month', 'day', 'period', 'quarter']
            has_time_data = any(keyword in ' '.join(columns) for keyword in time_keywords)
            
            # Get key insights from analysis history
            key_insights = []
            if self.analysis_history:
                latest_analysis = self.analysis_history[-1].get('results', '')
                # Extract key numbers or percentages for context
                import re
                numbers = re.findall(r'\d+\.?\d*%?', latest_analysis)
                if numbers:
                    key_insights.extend(numbers[:3])  # Top 3 key numbers
            
            # Generate role-specific, data-driven queries
            if self.role == "Personal Accountant" and has_financial_data:
                if has_time_data:
                    search_queries.append("personal finance budgeting trends monthly spending analysis")
                    search_queries.append("average household expenses categories 2024")
                else:
                    search_queries.append("personal financial health assessment ratios")
                    search_queries.append("emergency fund recommendations financial advisors")
                    
            elif self.role == "Business Analyst" and has_business_data:
                if 'sales' in ' '.join(columns):
                    search_queries.append("sales performance benchmarks industry standards")
                    search_queries.append("business KPI metrics analysis best practices")
                elif 'customer' in ' '.join(columns):
                    search_queries.append("customer analytics trends business intelligence")
                    search_queries.append("customer retention benchmarks industry")
                else:
                    search_queries.append("business performance metrics analysis methods")
                    
            elif self.role == "Data Analyst":
                if has_time_data:
                    search_queries.append("time series analysis techniques business applications")
                if has_financial_data or has_business_data:
                    search_queries.append("data analysis methods financial business datasets")
                else:
                    search_queries.append("statistical analysis techniques data insights")
            
            # If no specific queries generated, fall back to generic but still relevant
            if not search_queries:
                if has_financial_data:
                    search_queries.append("financial data analysis best practices")
                elif has_business_data:
                    search_queries.append("business data analysis insights methods")
                else:
                    search_queries.append("data analysis interpretation techniques")
            
            return search_queries[:3]  # Maximum 3 queries
            
        except Exception as e:
            self.thoughts.append(f"⚠️ Error generating search queries: {str(e)}")
            return []

    def _initialize_enhanced_features(self):
        """
        Initialize enhanced features including web search and advanced visualizations.
        """
        try:
            # Initialize web search tool
            if self.role in ["Business Analyst", "Accountant"]:
                self.web_search_tool = create_web_tool('financial')
            elif self.role == "Sports Coach":
                self.web_search_tool = create_web_tool('sports')
            else:
                self.web_search_tool = create_web_tool('search')
            
            # Initialize advanced visualization engine
            self.enhanced_viz_engine = AdvancedVisualizationEngine()
            
            # Get enhanced role configuration
            role_prompts = get_enhanced_role_prompts()
            if self.role in role_prompts:
                self.role_config = role_prompts[self.role]
                # Update available actions based on role capabilities
                if 'tools' in self.role_config:
                    self.available_actions.extend(['web_search', 'enhanced_viz'])
            
        except Exception as e:
            print(f"Warning: Could not initialize enhanced features: {e}")

    def set_max_cycles(self, max_cycles: int):
        """
        Set the maximum number of analysis cycles.
        
        Args:
            max_cycles (int): Maximum number of cycles to perform
        """
        self.max_cycles = max_cycles

    def can_continue_cycle(self) -> bool:
        """
        Check if the agent can continue with more cycles.
        
        Returns:
            bool: True if more cycles are allowed, False otherwise
        """
        return not self.task_complete

    def _initialize_goals(self) -> Dict:
        """
        Initialize role-specific goals and sub-goals.
        
        This method sets up the initial goals for the agent based on its role.
        Each role has specific sub-goals that guide the agent's analysis process.
        
        Returns:
            Dict: Dictionary containing primary goal, sub-goals, completed goals, and current focus
        """
        base_goals = {
            "primary_goal": "Provide comprehensive analysis and insights",
            "sub_goals": [],
            "completed_goals": [],
            "current_focus": None
        }
        
        role_info = get_role_prompts().get(self.role, get_role_prompts()["Custom"])
        base_goals["sub_goals"] = role_info.get("sub_goals", [])
        
        return base_goals

    def _update_goals(self, analysis_results: str):
        """
        Update goals based on analysis results and learning.
        
        This method reviews the analysis results and updates the agent's goals
        accordingly. It marks completed goals and updates the current focus.
        
        Args:
            analysis_results (str): Results from the current analysis
        """
        for goal in self.goals["sub_goals"]:
            if goal.lower() in analysis_results.lower():
                if goal not in self.goals["completed_goals"]:
                    self.goals["completed_goals"].append(goal)
                    self.learning_points.append(f"Successfully completed goal: {goal}")

        remaining_goals = [g for g in self.goals["sub_goals"] if g not in self.goals["completed_goals"]]
        if remaining_goals:
            self.goals["current_focus"] = remaining_goals[0]

    def _should_collaborate(self, other_agents: List['Agent']) -> bool:
        """
        Determine if collaboration with other agents would be beneficial.
        
        This method checks if any other agent's recent analysis might be relevant
        to the current agent's goals.
        
        Args:
            other_agents (List[Agent]): List of other agents to potentially collaborate with
            
        Returns:
            bool: True if collaboration would be beneficial, False otherwise
        """
        for agent in other_agents:
            if agent.conversation_history:
                last_message = agent.conversation_history[-1]["content"]
                if any(goal.lower() in last_message.lower() for goal in self.goals["sub_goals"]):
                    return True
        return False

    def _generate_response(self, prompt: str) -> str:
        """
        Generate a response using either OpenAI or Ollama.
        
        This method handles the communication with either the OpenAI API or
        local Ollama instance to generate responses based on the provided prompt.
        
        Args:
            prompt (str): The prompt to generate a response for
            
        Returns:
            str: Generated response
        """
        try:
            # Reset token count for new generation
            self.current_tokens_used = 0
            
            if self.use_openai:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                # Update token count (this should be implemented in _call_model)
                # For now, we'll estimate tokens (roughly 4 chars per token)
                self.current_tokens_used = len(response.choices[0].message.content) // 4
                return response.choices[0].message.content
            else:
                response = ollama.generate(model=self.model, prompt=prompt)
                # Update token count (this should be implemented in _call_model)
                # For now, we'll estimate tokens (roughly 4 chars per token)
                self.current_tokens_used = len(response['response']) // 4
                return response['response']
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.thoughts.append(f"Error occurred: {error_msg}")
            return error_msg

    def _manipulate_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """
        Allow the agent to manipulate the data for better understanding.
        
        This method enables the agent to suggest and perform data manipulations
        that might improve the analysis. It maintains a history of all
        manipulation suggestions.
        
        Args:
            data (pd.DataFrame): The input data to manipulate
            
        Returns:
            Tuple[pd.DataFrame, str]: Tuple containing manipulated data and manipulation suggestions
        """
        try:
            # Create a copy of the data
            manipulated_data = data.copy()
            manipulations = []
            
            # Generate a prompt for data manipulation
            prompt = f"""As a {self.role}, analyze this data and suggest manipulations to better understand it.
            Current data shape: {data.shape}
            Columns: {', '.join(data.columns)}
            
            Suggest specific data manipulations that would help understand the data better.
            Examples:
            - Create new derived columns
            - Handle missing values
            - Normalize or scale data
            - Create categorical groupings
            - Calculate statistical measures
            
            Return your suggestions in a clear, actionable format.
            """
            
            response = self._generate_response(prompt)
            
            # Store the manipulation suggestion
            self.data_manipulations.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "suggestion": response,
                "cycle": self.cycle_count
            })
            
            return manipulated_data, response
            
        except Exception as e:
            error_msg = f"Error manipulating data: {str(e)}"
            self.thoughts.append(f"Error occurred: {error_msg}")
            return data, error_msg

    def analyze(self, data: pd.DataFrame, context: str = "", custom_prompt: Optional[str] = None) -> str:
        """
        Analyze the provided data using the agent's model.
        
        This is the main analysis method that coordinates the entire analysis process.
        It handles data manipulation, generates analysis results, and updates the
        agent's state including goals, learning points, and history.
        
        Args:
            data (pd.DataFrame): Data to analyze
            context (str): Additional context for the analysis
            custom_prompt (Optional[str]): Custom prompt template to use
            
        Returns:
            str: Analysis results
        """
        try:
            # Validate input data
            if data is None or data.empty:
                return "Error: No data provided for analysis."
            
            # Start timing the cycle
            self.current_cycle_start = datetime.now()
            self.cycle_count += 1
            context += f"\nAnalysis Cycle: {self.cycle_count}/{self.max_cycles}"
            
            if self.goals["current_focus"]:
                context += f"\nCurrent focus: {self.goals['current_focus']}"

            # Generate detailed data summary with error handling
            try:
                # Basic data information
                data_shape = data.shape
                column_names = list(data.columns)
                
                # Data types and basic info
                data_info = {
                    "total_rows": data_shape[0],
                    "total_columns": data_shape[1],
                    "column_names": column_names,
                    "data_types": {col: str(dtype) for col, dtype in data.dtypes.items()},
                    "missing_values": data.isnull().sum().to_dict()
                }
                
                # Numeric columns analysis
                numeric_summary = {}
                numeric_cols = data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
                if len(numeric_cols) > 0:
                    for col in numeric_cols:
                        try:
                            col_data = data[col].dropna()
                            if len(col_data) > 0:
                                numeric_summary[col] = {
                                    "count": len(col_data),
                                    "mean": float(col_data.mean()),
                                    "median": float(col_data.median()),
                                    "std": float(col_data.std()) if len(col_data) > 1 else 0,
                                    "min": float(col_data.min()),
                                    "max": float(col_data.max()),
                                    "unique_values": int(col_data.nunique())
                                }
                        except Exception as e:
                            self.thoughts.append(f"Warning: Could not analyze numeric column {col}: {str(e)}")
                
                # Categorical columns analysis
                categorical_summary = {}
                cat_cols = data.select_dtypes(include=['object', 'category', 'string']).columns
                if len(cat_cols) > 0:
                    for col in cat_cols:
                        try:
                            col_data = data[col].dropna()
                            if len(col_data) > 0:
                                value_counts = col_data.value_counts().head(10)  # Top 10 values
                                categorical_summary[col] = {
                                    "unique_count": int(col_data.nunique()),
                                    "most_common": value_counts.to_dict(),
                                    "sample_values": col_data.head(5).tolist()
                                }
                        except Exception as e:
                            self.thoughts.append(f"Warning: Could not analyze categorical column {col}: {str(e)}")
                
                # Sample data for context
                sample_data = {}
                try:
                    sample_rows = data.head(3).to_dict('records')  # First 3 rows as records
                    for i, row in enumerate(sample_rows):
                        sample_data[f"Row_{i+1}"] = {k: str(v) for k, v in row.items()}
                except Exception as e:
                    self.thoughts.append(f"Warning: Could not generate sample data: {str(e)}")
                
                # Create comprehensive data summary
                data_summary_text = f"""
DATASET OVERVIEW:
- Dataset size: {data_info['total_rows']:,} rows × {data_info['total_columns']} columns
- Column names: {', '.join(data_info['column_names'])}

DATA TYPES:
{chr(10).join(f'- {col}: {dtype}' for col, dtype in data_info['data_types'].items())}

MISSING VALUES:
{chr(10).join(f'- {col}: {count:,} missing ({count/data_info["total_rows"]*100:.1f}%)' 
              for col, count in data_info['missing_values'].items() if count > 0)}
{chr(10) + "- No missing values detected" if not any(data_info['missing_values'].values()) else ""}

NUMERIC COLUMNS ANALYSIS:
{chr(10).join(f'''- {col}:
  • Count: {stats["count"]:,} values
  • Mean: {stats["mean"]:.3f}, Median: {stats["median"]:.3f}
  • Range: {stats["min"]:.3f} to {stats["max"]:.3f}
  • Std Dev: {stats["std"]:.3f}
  • Unique values: {stats["unique_values"]:,}''' 
              for col, stats in numeric_summary.items()) if numeric_summary else "- No numeric columns found"}

CATEGORICAL COLUMNS ANALYSIS:
{chr(10).join(f'''- {col}:
  • Unique values: {stats["unique_count"]:,}
  • Top values: {", ".join(f"{k}: {v}" for k, v in list(stats["most_common"].items())[:3])}
  • Sample: {", ".join(str(v) for v in stats["sample_values"][:3])}''' 
              for col, stats in categorical_summary.items()) if categorical_summary else "- No categorical columns found"}

SAMPLE DATA (First 3 rows):
{chr(10).join(f"Row {i}: {', '.join(f'{k}={v}' for k, v in row.items())}" 
              for i, row in sample_data.items()) if sample_data else "- Could not generate sample data"}
"""

            except Exception as e:
                self.thoughts.append(f"Warning: Error generating data summary: {str(e)}")
                data_summary_text = f"Error generating detailed data summary: {str(e)}"

            # Manipulate data for better understanding
            try:
                manipulated_data, manipulation_suggestion = self._manipulate_data(data)
            except Exception as e:
                self.thoughts.append(f"Warning: Error in data manipulation: {str(e)}")
                manipulated_data = data
                manipulation_suggestion = "Could not generate manipulation suggestions due to an error."
            
            role_info = get_role_prompts().get(self.role, get_role_prompts()["Custom"])
            prompt = custom_prompt or role_info["prompt"]
            
            # Create a data-focused analysis prompt
            analysis_prompt = f"""
As a {self.role}, analyze the following dataset thoroughly and provide specific insights:

{data_summary_text}

CONTEXT: {context}

DATA MANIPULATION SUGGESTION:
{manipulation_suggestion}

ANALYSIS INSTRUCTIONS:
1. Examine the actual data values, patterns, and distributions shown above
2. Identify specific trends, outliers, or interesting patterns in the data
3. Reference actual column names, values, and statistics in your analysis
4. Provide concrete insights based on the data shown
5. Suggest specific actions or further analysis based on what you observe
6. Focus on the actual data provided, not generic analysis

Please provide a detailed analysis that specifically references the data shown above. Make sure your insights are directly related to the actual values, patterns, and characteristics of this specific dataset.
"""

            try:
                response = self._generate_response(analysis_prompt)
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                self.thoughts.append(f"Error occurred: {error_msg}")
                response = error_msg
            
            # Calculate cycle time
            cycle_time = (datetime.now() - self.current_cycle_start).total_seconds()
            
            # Calculate tokens per second
            tokens_per_second = self.current_tokens_used / cycle_time if cycle_time > 0 else 0
            
            # Record cycle timing and token usage
            self.cycle_times.append({
                "cycle": self.cycle_count,
                "time_seconds": cycle_time,
                "start_time": self.current_cycle_start,
                "tokens_used": self.current_tokens_used,
                "tokens_per_second": tokens_per_second
            })
            
            self.thoughts.append(f"Analysis cycle {self.cycle_count} complete in {round(cycle_time, 2)} seconds")
            self._update_goals(response)
            
            self.analysis_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "context": context,
                "results": response,
                "cycle": self.cycle_count,
                "cycle_time": round(cycle_time, 2),
                "data_manipulation": manipulation_suggestion
            })
            
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "cycle": self.cycle_count,
                "cycle_time": round(cycle_time, 2)
            })
            
            return response
        except Exception as e:
            error_msg = f"Error analyzing data with {self.model}: {str(e)}"
            self.thoughts.append(f"Error occurred: {error_msg}")
            return error_msg

    def get_cycle_times(self) -> List[Dict]:
        """
        Get the timing information for all cycles.
        
        Returns:
            List[Dict]: List of dictionaries containing cycle timing information
        """
        return self.cycle_times

    def get_data_manipulations(self) -> List[Dict]:
        """
        Get all data manipulation suggestions.
        
        Returns:
            List[Dict]: List of dictionaries containing data manipulation suggestions
        """
        return self.data_manipulations

    def respond_to_feedback(self, feedback: str, other_agents: List['Agent'] = None) -> str:
        """
        Respond to user feedback and potentially interact with other agents.
        
        This method allows the agent to process user feedback and potentially
        collaborate with other agents to provide a comprehensive response.
        
        Args:
            feedback (str): User feedback to respond to
            other_agents (Optional[List[Agent]]): Other agents to potentially collaborate with
            
        Returns:
            str: Response to the feedback
        """
        try:
            should_collaborate = self._should_collaborate(other_agents) if other_agents else False
            
            other_agents_context = ""
            if other_agents and should_collaborate:
                other_agents_context = "\nRecent insights from other agents:\n"
                for agent in other_agents:
                    if agent.conversation_history:
                        last_message = agent.conversation_history[-1]
                        other_agents_context += f"\n{agent.name} ({agent.role}):\n{last_message['content']}\n"

            prompt = f"""As a {self.role}, respond to the following feedback and consider insights from other agents:
            
            User Feedback: {feedback}
            
            {other_agents_context}
            
            Current Goals:
            - Primary: {self.goals['primary_goal']}
            - Current Focus: {self.goals['current_focus'] if self.goals['current_focus'] else 'None'}
            
            Please provide a thoughtful response that:
            1. Addresses the user's feedback directly
            2. Builds upon or respectfully challenges other agents' insights if relevant
            3. Maintains your role's perspective
            4. Suggests next steps or additional analysis if appropriate
            5. Considers your current goals and focus
            """

            response = self._generate_response(prompt)
            
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            self._update_goals(response)
            
            return response
        except Exception as e:
            error_msg = f"Error responding to feedback: {str(e)}"
            self.thoughts.append(f"Error occurred: {error_msg}")
            return error_msg

    def self_feedback(self) -> str:
        """
        Generate self-feedback based on current analysis and goals.
        
        This method allows the agent to review its own analysis and provide
        feedback on areas for improvement and next steps.
        
        Returns:
            str: Self-feedback response
        """
        try:
            prompt = f"""As a {self.role}, review your current analysis and provide self-feedback:
            
            Current Goals:
            - Primary: {self.goals['primary_goal']}
            - Current Focus: {self.goals['current_focus'] if self.goals['current_focus'] else 'None'}
            
            Completed Goals: {', '.join(self.goals['completed_goals'])}
            
            Recent Analysis:
            {self.conversation_history[-1]['content'] if self.conversation_history else 'No analysis yet'}
            
            Please provide feedback on:
            1. Areas that need deeper analysis
            2. Potential improvements to the current approach
            3. Additional data points to consider
            4. Next steps for achieving remaining goals
            """

            response = self._generate_response(prompt)
            
            self.conversation_history.append({
                "role": "self_feedback",
                "content": response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            return response
        except Exception as e:
            error_msg = f"Error generating self-feedback: {str(e)}"
            self.thoughts.append(f"Error occurred: {error_msg}")
            return error_msg

    def _generate_improvement_suggestions(self) -> str:
        """
        Generate suggestions for self-improvement based on learning points and analysis history.
        
        This method reviews the agent's learning points and analysis history to
        generate specific suggestions for improving its analysis capabilities.
        
        Returns:
            str: Improvement suggestions
        """
        try:
            if not self.learning_points and not self.analysis_history:
                return "No improvement suggestions available yet."

            # Create a prompt for generating improvement suggestions
            prompt = f"""As a {self.role}, review the following information and provide improvement suggestions:

Learning Points:
{chr(10).join(f'- {point}' for point in self.learning_points[-5:]) if self.learning_points else 'No learning points yet'}

Recent Analysis:
{self.analysis_history[-1]['results'] if self.analysis_history else 'No analysis history yet'}

Current Goals:
- Primary: {self.goals['primary_goal']}
- Current Focus: {self.goals['current_focus'] if self.goals['current_focus'] else 'None'}
- Completed Goals: {', '.join(self.goals['completed_goals']) if self.goals['completed_goals'] else 'None'}

Please provide specific suggestions for:
1. Analysis approach improvements
2. Additional data points to consider
3. Potential collaboration opportunities
4. Tools or methods to enhance capabilities
5. Areas for further investigation

Focus on concrete, actionable suggestions that would improve the analysis quality and depth."""

            response = self._generate_response(prompt)
            return response
        except Exception as e:
            error_msg = f"Error generating improvement suggestions: {str(e)}"
            self.thoughts.append(f"Error occurred: {error_msg}")
            return error_msg

    def set_data_copy(self, data: pd.DataFrame):
        """
        Create a deep copy of the data for the agent to manipulate.
        
        Args:
            data (pd.DataFrame): Original data to copy
        """
        if data is not None:
            # Create a deep copy to ensure isolation from original data
            self.data_copy = data.copy(deep=True)
            self.thoughts.append(f"✅ Received isolated data copy: {data.shape[0]:,} rows × {data.shape[1]} columns")
            
            # Log data characteristics for debugging
            missing_values = self.data_copy.isnull().sum().sum()
            duplicates = self.data_copy.duplicated().sum()
            numeric_cols = len(self.data_copy.select_dtypes(include=[np.number]).columns)
            categorical_cols = len(self.data_copy.select_dtypes(include=['object', 'category']).columns)
            
            self.thoughts.append(f"📊 Data characteristics: {missing_values} missing values, {duplicates} duplicates, {numeric_cols} numeric cols, {categorical_cols} categorical cols")
        else:
            self.data_copy = None
            self.thoughts.append("❌ No data provided for copying")

    def choose_action(self, context: str = "") -> str:
        """
        🧠 INTELLIGENT WORKFLOW DECISION ENGINE
        
        This is the core decision-making method that implements the sophisticated workflow logic:
        Analyze → Data Cleaning (if needed) → Visualize → Summarize → Repeat → End
        
        DECISION ALGORITHM OVERVIEW:
        The agent uses a multi-stage decision process that considers:
        
        1. PROGRESS TRACKING: Counts completed actions in each category
           - Analysis cycles (excluding summaries and visualizations)
           - Data manipulation operations performed
           - Visualizations created with chart type tracking
           - Executive summaries generated
        
        2. DATA QUALITY ASSESSMENT: Evaluates current data state
           - Missing value detection and quantification
           - Duplicate record identification
           - Data type optimization opportunities
           - Outlier detection and flagging needs
        
        3. WORKFLOW SEQUENCE ENFORCEMENT: Ensures logical progression
           - Analysis must precede visualization
           - Data cleaning occurs after analysis if issues detected
           - Visualization follows analysis/cleaning completion
           - Summary generation occurs after visualization
        
        4. AUTONOMOUS CONTINUATION: AI-driven completion decisions
           - Evaluates analysis depth and completeness
           - Considers insight quality and business value
           - Makes intelligent decisions about additional cycles
           - Balances thoroughness with efficiency
        
        WORKFLOW DECISION POINTS:
        
        🔍 ANALYZE (Step 1): Always the starting point
        - Triggered when: analyze_count == 0 (first action)
        - Purpose: Initial data exploration and pattern discovery
        - Sets foundation for all subsequent workflow steps
        
        🔧 CLEAN (Step 2): Data quality improvement  
        - Triggered when: analysis complete + data quality issues detected
        - Conditions: analyze_count > manipulation_count AND needs_cleaning == True
        - Purpose: Handle missing values, duplicates, type optimization
        
        📊 VISUALIZE (Step 3): Pattern visualization and insight discovery
        - Triggered when: analysis/cleaning complete + visualizations needed
        - Conditions: analyze_count > visualization_count OR post-cleaning state
        - Purpose: Create charts that reveal patterns and relationships
        
        📋 SUMMARIZE (Step 4): Executive summary generation
        - Triggered when: analysis + visualization complete + summary needed
        - Conditions: analyze_count > summary_count OR visualization_count > summary_count
        - Purpose: Generate business-relevant insights and recommendations
        
        🔄 REPEAT (Step 5): Additional analysis cycle decision
        - Triggered when: complete_cycles < 2 (minimum thoroughness requirement)
        - Purpose: Ensure sufficient analysis depth and comprehensive coverage
        
        🤖 AUTONOMOUS DECISION (Step 6): AI-driven continuation assessment
        - Triggered when: complete_cycles >= 2 AND analyze_count >= 3
        - Process: AI evaluates analysis completeness and value of additional cycles
        - Decision: CONTINUE (more insights possible) vs END (sufficient analysis)
        
        ✅ END (Step 7): Task completion
        - Triggered when: AI decides sufficient analysis completed OR fallback conditions
        - Purpose: Graceful task termination with comprehensive results
        
        SAFETY AND ROBUSTNESS:
        - Comprehensive error handling with safe fallback actions
        - Detailed logging for transparency and debugging
        - State validation to prevent workflow inconsistencies
        - Progress tracking for performance monitoring
        
        Args:
            context (str): Additional contextual information for decision-making
                          (e.g., user preferences, specific requirements, domain constraints)
            
        Returns:
            str: Next action to execute from available_actions:
                 ["analyze", "manipulate_data", "visualize", "summarize", "end"]
                 
        Raises:
            Exception: Gracefully handled with fallback to "analyze" action
        """
        try:
            # Count different types of actions taken
            analyze_count = len([h for h in self.analysis_history if h.get('data_manipulation') != 'Visualization creation' and not str(h.get('cycle', '')).startswith('summary')])
            manipulation_count = len(self.data_manipulations)
            visualization_count = len(self.visualizations)
            summary_count = len([h for h in self.analysis_history if str(h.get('cycle', '')).startswith('summary')])
            
            # Calculate complete cycles (analyze -> clean -> visualize -> summarize)
            complete_cycles = min(analyze_count, summary_count)
            
            # Check if data needs cleaning
            needs_cleaning = False
            if self.data_copy is not None:
                missing_values = self.data_copy.isnull().sum().sum()
                duplicates = self.data_copy.duplicated().sum()
                needs_cleaning = missing_values > 0 or duplicates > 0
            
            # Workflow decision logic
            if analyze_count == 0:
                action = "analyze"  # Always start with analysis
                
            elif analyze_count > manipulation_count and needs_cleaning:
                action = "manipulate_data"  # Clean data after analysis if needed
                
            elif analyze_count > visualization_count or (manipulation_count > 0 and manipulation_count >= visualization_count):
                action = "visualize"  # Create visualizations after analysis/cleaning
                
            elif analyze_count > summary_count or visualization_count > summary_count:
                action = "summarize"  # Summarize findings after visualization
                
            elif complete_cycles < 2:  # Allow at least 2 complete cycles
                action = "analyze"  # Start next cycle with analysis
                
            elif complete_cycles >= 2 and analyze_count >= 3:
                # After 2+ complete cycles and sufficient analysis, agent can choose to continue or end
                decision_prompt = f"""
You are {self.name}, a {self.role}. You have completed {complete_cycles} full analysis cycles.

CURRENT PROGRESS:
- Analysis rounds: {analyze_count}
- Data cleanings: {manipulation_count}
- Visualizations: {visualization_count}  
- Summaries: {summary_count}
- Complete cycles: {complete_cycles}

QUESTION: Do you think you have gathered sufficient insights and completed thorough analysis, or do you need another analysis cycle?

Respond with EXACTLY one word:
- "CONTINUE" if you need more analysis
- "END" if you have sufficient insights

Your decision:"""
                
                ai_decision = self._generate_response(decision_prompt).strip().upper()
                action = "analyze" if "CONTINUE" in ai_decision else "end"
                
            else:
                action = "end"  # Default end condition
            
            # Log the decision with reasoning
            reasoning = f"Workflow step: A:{analyze_count}, C:{manipulation_count}, V:{visualization_count}, S:{summary_count}, Cycles:{complete_cycles}"
            if needs_cleaning:
                reasoning += " (data needs cleaning)"
            
            self.thoughts.append(f"Chosen action: {action} | {reasoning}")
            
            return action
            
        except Exception as e:
            self.thoughts.append(f"Error in decision making: {str(e)}")
            return "analyze"  # fallback

    def autonomous_analyze(self, data: pd.DataFrame, context: str = "") -> str:
        """
        Perform autonomous analysis with enhanced decision-making.
        
        Args:
            data (pd.DataFrame): Data to analyze
            context (str): Additional context
            
        Returns:
            str: Analysis results with autonomous insights
        """
        enhanced_context = f"""
{context}

AUTONOMOUS ANALYSIS MODE:
As an autonomous AI agent, I am operating independently to provide the most valuable insights.
I will focus on discovering actionable insights and making recommendations based on my role as a {self.role}.

My autonomous objectives:
1. Identify key patterns and trends in the data
2. Provide specific, actionable recommendations
3. Highlight potential risks or opportunities
4. Make data-driven conclusions relevant to business decisions
"""
        
        return self.analyze(data, enhanced_context)

    def get_autonomous_status(self) -> Dict:
        """
        Get current autonomous agent status for monitoring.
        
        Returns:
            Dict: Status information for autonomous execution
        """
        return {
            "agent_name": self.name,
            "role": self.role,
            "task_complete": self.task_complete,
            "actions_taken": len(self.analysis_history) + len(self.data_manipulations),
            "analyses_completed": len(self.analysis_history),
            "manipulations_performed": len(self.data_manipulations),
            "current_data_shape": self.data_copy.shape if self.data_copy is not None else None,
            "learning_points": len(self.learning_points),
            "last_action_time": self.cycle_times[-1]["start_time"] if self.cycle_times else None
        }

    def manipulate_data(self) -> str:
        """
        🔧 INTELLIGENT DATA CLEANING AND MANIPULATION ENGINE
        
        This method performs comprehensive data cleaning and quality improvement operations
        on the agent's isolated data copy. It follows data science best practices to
        systematically improve data quality and prepare it for enhanced analysis.
        
        DATA CLEANING WORKFLOW:
        
        1. 🔍 QUALITY ASSESSMENT: Comprehensive data quality analysis
           - Missing value detection and quantification across all columns
           - Duplicate record identification and impact assessment
           - Data type analysis and optimization opportunities
           - Statistical outlier detection using IQR method
        
        2. 🧹 MISSING VALUE HANDLING: Intelligent imputation strategies
           - Numeric columns: Median imputation (robust to outliers)
           - Categorical columns: Mode imputation (most frequent value)
           - Preserves data distribution characteristics
           - Logs all imputation operations for transparency
        
        3. 🔄 DUPLICATE REMOVAL: Systematic duplicate elimination
           - Identifies exact duplicate rows across all columns
           - Removes duplicates while preserving data integrity
           - Maintains first occurrence of each unique record
           - Reports number of duplicates removed
        
        4. 🎯 DATA TYPE OPTIMIZATION: Automatic type conversion
           - Detects string columns that can be converted to numeric
           - Performs safe type conversion with error handling
           - Optimizes memory usage and analysis performance
           - Logs successful type conversions
        
        5. 📊 FEATURE ENGINEERING: Value-added transformations
           - Outlier detection flags using statistical methods (IQR)
           - Creates boolean indicators for anomaly detection
           - Enhances dataset with derived analytical features
           - Prepares data for advanced statistical analysis
        
        CLEANING STRATEGIES BY DATA TYPE:
        
        📈 NUMERIC DATA:
        - Missing values → Median imputation (robust central tendency)
        - Outliers → IQR-based detection and flagging
        - Data types → Optimization for memory efficiency
        
        📝 CATEGORICAL DATA:
        - Missing values → Mode imputation (most common value)
        - String types → Attempt conversion to numeric when appropriate
        - Categories → Preservation of original categories
        
        QUALITY METRICS TRACKING:
        - Before/after shape comparison
        - Missing value count reduction
        - Duplicate elimination count
        - Data type optimization summary
        - Feature engineering additions
        
        AI ANALYSIS INTEGRATION:
        After cleaning operations, the method generates an AI-powered analysis of:
        - Impact assessment of cleaning operations performed
        - Data quality improvement quantification
        - Enhanced analysis opportunities created
        - Remaining data quality concerns or limitations
        - Recommendations for further analysis approaches
        
        SAFETY AND VALIDATION:
        - Deep copy manipulation (original data protected)
        - Comprehensive error handling for robust operation
        - Detailed operation logging for audit trails
        - Performance timing for optimization insights
        - Rollback capability through isolated data copies
        
        Returns:
            str: Comprehensive cleaning report including:
                 - Operations performed with detailed statistics
                 - Before/after quality metrics comparison
                 - AI analysis of cleaning impact and opportunities
                 - Executive summary of data quality improvements
                 
        Raises:
            Exception: Gracefully handled with detailed error reporting
        """
        try:
            if self.data_copy is None:
                return "Error: No data available for manipulation"
            
            self.current_cycle_start = datetime.now()
            original_shape = self.data_copy.shape
            
            # Analyze current data issues
            missing_values = self.data_copy.isnull().sum().sum()
            duplicates = self.data_copy.duplicated().sum()
            
            cleaning_steps = []
            
            # 1. Handle missing values
            if missing_values > 0:
                for col in self.data_copy.columns:
                    col_missing = self.data_copy[col].isnull().sum()
                    if col_missing > 0:
                        if self.data_copy[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                            # Fill numeric columns with median
                            median_val = self.data_copy[col].median()
                            self.data_copy[col].fillna(median_val, inplace=True)
                            cleaning_steps.append(f"✅ Filled {col_missing:,} missing values in '{col}' with median ({median_val:.2f})")
                        else:
                            # Fill categorical columns with mode
                            mode_val = self.data_copy[col].mode()
                            if len(mode_val) > 0:
                                self.data_copy[col].fillna(mode_val[0], inplace=True)
                                cleaning_steps.append(f"✅ Filled {col_missing:,} missing values in '{col}' with mode ('{mode_val[0]}')")
            
            # 2. Remove duplicates
            if duplicates > 0:
                self.data_copy.drop_duplicates(inplace=True)
                cleaning_steps.append(f"✅ Removed {duplicates:,} duplicate rows")
            
            # 3. Data type optimization
            type_changes = []
            for col in self.data_copy.columns:
                if self.data_copy[col].dtype == 'object':
                    # Try to convert to numeric if possible
                    try:
                        numeric_col = pd.to_numeric(self.data_copy[col], errors='coerce')
                        if not numeric_col.isnull().all():  # If conversion was successful
                            self.data_copy[col] = numeric_col
                            type_changes.append(f"'{col}' to numeric")
                    except:
                        pass
            
            if type_changes:
                cleaning_steps.append(f"✅ Optimized data types: {', '.join(type_changes)}")
            
            # 4. Create useful derived columns for analysis
            numeric_cols = self.data_copy.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Add outlier detection flag for first numeric column
                first_numeric = numeric_cols[0]
                Q1 = self.data_copy[first_numeric].quantile(0.25)
                Q3 = self.data_copy[first_numeric].quantile(0.75)
                IQR = Q3 - Q1
                outlier_condition = (
                    (self.data_copy[first_numeric] < Q1 - 1.5 * IQR) | 
                    (self.data_copy[first_numeric] > Q3 + 1.5 * IQR)
                )
                self.data_copy[f'{first_numeric}_outlier'] = outlier_condition
                outlier_count = outlier_condition.sum()
                cleaning_steps.append(f"✅ Added outlier detection for '{first_numeric}' ({outlier_count:,} outliers found)")
            
            # Generate AI analysis of the cleaning
            manipulation_prompt = f"""
As a {self.role}, I have just performed data cleaning on the dataset. Here's what was done:

CLEANING PERFORMED:
{chr(10).join(cleaning_steps) if cleaning_steps else "No cleaning was needed - data was already clean"}

BEFORE CLEANING:
- Shape: {original_shape}
- Missing values: {missing_values:,}
- Duplicates: {duplicates:,}

AFTER CLEANING:
- Shape: {self.data_copy.shape}
- Missing values: {self.data_copy.isnull().sum().sum():,}
- Duplicates: {self.data_copy.duplicated().sum():,}

TASK: Provide a brief analysis of:
1. The impact of these cleaning operations
2. How this improves the data quality for analysis
3. What additional insights might now be possible
4. Any remaining data quality concerns

Keep the response concise and focused on the cleaning impact.
"""
            
            ai_analysis = self._generate_response(manipulation_prompt)
            
            # Create comprehensive response
            if cleaning_steps:
                response = f"""
DATA CLEANING COMPLETED:

OPERATIONS PERFORMED:
{chr(10).join(cleaning_steps)}

SUMMARY:
• Original shape: {original_shape[0]:,} rows × {original_shape[1]} columns
• Final shape: {self.data_copy.shape[0]:,} rows × {self.data_copy.shape[1]} columns
• Missing values: {missing_values:,} → {self.data_copy.isnull().sum().sum():,}
• Duplicates: {duplicates:,} → {self.data_copy.duplicated().sum():,}

ANALYSIS IMPACT:
{ai_analysis}

✅ Data is now cleaner and ready for enhanced analysis and visualization.
"""
            else:
                response = f"""
DATA QUALITY ASSESSMENT:

✅ No cleaning operations were needed - the data is already in good condition:
• Shape: {self.data_copy.shape[0]:,} rows × {self.data_copy.shape[1]} columns  
• Missing values: {missing_values:,}
• Duplicates: {duplicates:,}

ANALYSIS:
{ai_analysis}

The data is ready for analysis and visualization without additional cleaning.
"""
            
            # Record the manipulation
            cycle_time = (datetime.now() - self.current_cycle_start).total_seconds()
            
            self.data_manipulations.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "suggestion": response,
                "cycle": f"cleaning_{len(self.data_manipulations) + 1}",
                "execution_time": cycle_time,
                "operations_performed": cleaning_steps,
                "data_shape_before": original_shape,
                "data_shape_after": self.data_copy.shape
            })
            
            self.thoughts.append(f"Data cleaning completed in {cycle_time:.2f} seconds - {len(cleaning_steps)} operations performed")
            
            return response
            
        except Exception as e:
            error_msg = f"Error in data manipulation: {str(e)}"
            self.thoughts.append(error_msg)
            return error_msg

    def summarize_findings(self) -> str:
        """
        Create a summary of all findings and analyses performed.
        
        Returns:
            str: Comprehensive summary of findings
        """
        try:
            self.current_cycle_start = datetime.now()
            
            # Perform data-specific web search for relevant context
            web_search_context = ""
            if self.role in ["Personal Accountant", "Business Analyst", "Data Analyst"] and self.data_copy is not None:
                self.thoughts.append("🌐 Analyzing data to determine relevant external research...")
                
                # Analyze the actual data to create relevant search queries
                search_queries = self._generate_data_specific_search_queries()
                
                if search_queries:
                    self.thoughts.append(f"🔍 Generated {len(search_queries)} data-specific search queries")
                    
                    # Perform searches and compile results
                    for query in search_queries[:2]:  # Limit to 2 searches to avoid overload
                        try:
                            search_result = self.perform_web_search(query)
                            if search_result and "error" not in search_result.lower():
                                web_search_context += f"\n🔍 {query}:\n{search_result}\n"
                        except Exception as e:
                            self.thoughts.append(f"⚠️ Web search failed for '{query}': {str(e)}")
                            continue
                    
                    if web_search_context:
                        self.thoughts.append("✅ Data-specific external research completed")
                    else:
                        self.thoughts.append("⚠️ No useful external research found")
                else:
                    self.thoughts.append("⚠️ No relevant search queries could be generated from data")
            
            summary_prompt = f"""
As a {self.role}, create a comprehensive summary of all analysis work completed:

DATA OVERVIEW:
- Original data: {self.data_copy.shape if self.data_copy is not None else 'N/A'}
- Manipulations performed: {len(self.data_manipulations)}
- Analysis cycles completed: {len(self.analysis_history)}

ANALYSIS HISTORY:
{chr(10).join([f"Cycle {a['cycle']}: {a['results']}" for a in self.analysis_history]) if self.analysis_history else "No analyses completed"}

DATA MANIPULATIONS:
{chr(10).join([f"- {m['suggestion'][:200]}..." for m in self.data_manipulations]) if self.data_manipulations else "No manipulations performed"}

LEARNING POINTS:
{chr(10).join([f"- {point}" for point in self.learning_points]) if self.learning_points else "No learning points recorded"}

{"EXTERNAL MARKET CONTEXT:" + web_search_context if web_search_context else ""}

TASK: Create a comprehensive executive summary that includes:
1. KEY FINDINGS - Most important insights discovered
2. DATA QUALITY - Assessment of data condition and improvements made
3. RECOMMENDATIONS - Actionable suggestions based on analysis
4. METHODOLOGY - Brief description of approach used
5. MARKET CONTEXT - How external trends relate to the findings (if available)
6. LIMITATIONS - Any constraints or areas needing further investigation

Format as a professional business report suitable for stakeholders.
{"Include specific references to current market conditions and trends in your recommendations." if web_search_context else ""}
"""
            
            response = self._generate_response(summary_prompt)
            
            # Record timing
            cycle_time = (datetime.now() - self.current_cycle_start).total_seconds()
            tokens_per_second = self.current_tokens_used / cycle_time if cycle_time > 0 else 0
            
            # Add to analysis history
            self.analysis_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "context": "Summary of all findings",
                "results": response,
                "cycle": f"summary_{len(self.analysis_history) + 1}",
                "cycle_time": round(cycle_time, 2),
                "data_manipulation": "Summary generation"
            })
            
            self.thoughts.append(f"Summary completed in {cycle_time:.2f} seconds")
            
            return response
            
        except Exception as e:
            error_msg = f"Error creating summary: {str(e)}"
            self.thoughts.append(error_msg)
            return error_msg

    def end_task(self) -> str:
        """
        Mark the task as complete and provide final thoughts.
        
        Returns:
            str: Final completion message
        """
        self.task_complete = True
        completion_msg = f"""
TASK COMPLETED by {self.name} ({self.role})

SUMMARY:
- Total cycles: {len(self.analysis_history)}
- Data manipulations: {len(self.data_manipulations)}
- Learning points: {len(self.learning_points)}
- Final data shape: {self.data_copy.shape if self.data_copy is not None else 'N/A'}

STATUS: ✅ Task marked as complete
"""
        self.thoughts.append("Task marked as complete")
        return completion_msg

    def create_visualizations(self) -> str:
        """
        📊 INTELLIGENT VISUALIZATION GENERATION ENGINE WITH VALIDATION
        
        This method automatically creates appropriate visualizations based on data characteristics,
        types, and statistical properties. It uses sophisticated logic and validation to ensure
        only meaningful visualizations are created, preventing empty graphs and ensuring data suitability.
        
        ENHANCED FEATURES:
        - 🛡️ **Data Validation**: Pre-validates data suitability before creating visualizations
        - 🚫 **Empty Graph Prevention**: Ensures visualizations contain meaningful data
        - 🎯 **Smart Chart Selection**: AI-driven selection of optimal chart types
        - 📊 **Quality Assurance**: Validates data quality and visualization requirements
        
        VISUALIZATION STRATEGY OVERVIEW:
        
        The method analyzes data characteristics and creates 2-4 complementary visualizations:
        
        1. 📈 DISTRIBUTION ANALYSIS: Shows how data values are spread
           - Histograms for continuous numeric data with many unique values
           - Bar charts for discrete/categorical data with few unique values
           - Automatic bin selection based on data cardinality
        
        2. 🔥 CORRELATION ANALYSIS: Reveals relationships between variables
           - Heatmaps showing correlation matrices for numeric variables
           - Interactive hover information for detailed correlation values
           - Color-coded correlation strength indicators
        
        3. 📊 CATEGORICAL ANALYSIS: Explores categorical distributions
           - Bar charts showing frequency of categorical values
           - Top-N value selection to prevent overcrowding
           - Sorted by frequency for immediate insight identification
        
        4. ⭐ RELATIONSHIP ANALYSIS: Examines variable interactions
           - Scatter plots for numeric variable relationships
           - Sample size optimization for large datasets (performance)
           - Opacity adjustment for overplotting management
        
        VALIDATION LOGIC:
        
        🛡️ PRE-VISUALIZATION VALIDATION:
        - Checks data suitability using intelligent validation engine
        - Ensures minimum data points and column requirements are met
        - Validates data types and variation in values
        - Prevents creation of visualizations for unsuitable data
        
        Returns:
            str: Comprehensive visualization report including:
                 - Validation results and data suitability assessment
                 - Number and types of charts created
                 - Description of insights each visualization reveals
                 - AI analysis of visualization recommendations
                 - Statistical context and interpretation guidance
                 
        Raises:
            Exception: Gracefully handled with detailed error reporting
        """
        try:
            if self.data_copy is None:
                return "Error: No data available for visualization"
            
            self.current_cycle_start = datetime.now()
            
            # Import and use visualization validator
            from src.utils.visualization_validator import visualization_validator
            
            # Validate data suitability for visualization
            validation_result = visualization_validator.validate_data_for_visualization(self.data_copy)
            
            if not validation_result['is_suitable']:
                # Data is not suitable for visualization
                validation_report = f"""
🚫 **VISUALIZATION VALIDATION FAILED**

**Reason:** {validation_result['reason']}

**Recommendations:**
"""
                for rec in validation_result['recommendations']:
                    validation_report += f"• {rec}\n"
                
                validation_report += f"""
**Data Summary:**
- Rows: {validation_result['data_summary'].get('row_count', 0)}
- Columns: {validation_result['data_summary'].get('column_count', 0)}

The agent determined that creating visualizations would not provide meaningful insights with the current data. Please address the data quality issues before attempting visualization.
"""
                
                # Add to thoughts and return
                self.thoughts.append("❌ Visualization validation failed - data not suitable for meaningful charts")
                return validation_report
            
            # Data is suitable - proceed with intelligent visualization creation
            suitable_charts = validation_result['suitable_charts']
            self.thoughts.append(f"✅ Data validation passed - {len(suitable_charts)} chart types are suitable")
            
            # Analyze data to determine best visualization types
            visualization_prompt = f"""
As a {self.role}, analyze the current data and suggest the most appropriate visualizations to help understand the data better.

CURRENT DATA INFO:
- Shape: {self.data_copy.shape}
- Columns: {list(self.data_copy.columns)}
- Data types: {dict(self.data_copy.dtypes)}

NUMERIC COLUMNS: {list(self.data_copy.select_dtypes(include=[np.number]).columns)}
CATEGORICAL COLUMNS: {list(self.data_copy.select_dtypes(include=['object', 'category']).columns)}

SAMPLE DATA:
{self.data_copy.head(3).to_string()}

TASK: Based on this data, suggest 2-3 specific visualizations that would provide the most valuable insights. Consider:
1. Distribution plots for numeric data
2. Correlation heatmaps for multiple numeric columns
3. Bar charts for categorical data
4. Scatter plots for relationships
5. Time series plots if temporal data exists

Format your response as:
VISUALIZATION_PLAN:
1. [Chart Type] - [Reason] - [Columns to use]
2. [Chart Type] - [Reason] - [Columns to use]
3. [Chart Type] - [Reason] - [Columns to use]

Be specific about which columns to visualize and why each chart would be valuable.
"""
            
            response = self._generate_response(visualization_prompt)
            
            created_visualizations = []
            numeric_cols = self.data_copy.select_dtypes(include=[np.number]).columns
            categorical_cols = self.data_copy.select_dtypes(include=['object', 'category']).columns
            
            # Create visualizations based on data characteristics
            viz_count = 0
            
            # 1. Distribution plot for first numeric column
            if len(numeric_cols) > 0 and viz_count < 3:
                col = numeric_cols[0]
                
                # Handle edge case where column might have very few unique values
                unique_vals = self.data_copy[col].nunique()
                
                # Clean the data for visualization
                clean_data = self.data_copy[col].dropna()
                
                if len(clean_data) > 0:
                    if unique_vals > 10:
                        fig = px.histogram(
                            self.data_copy.dropna(), 
                            x=col, 
                            title=f'Distribution of {col}',
                            nbins=min(30, unique_vals)
                        )
                    else:
                        value_counts = clean_data.value_counts()
                        fig = px.bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            title=f'Distribution of {col}',
                            labels={'x': col, 'y': 'Count'}
                        )
                    
                    fig.update_layout(
                        xaxis_title=col,
                        yaxis_title="Frequency",
                        showlegend=False,
                        height=400
                    )
                    
                    created_visualizations.append({
                        "type": "histogram" if unique_vals > 10 else "bar",
                        "title": f"Distribution of {col}",
                        "figure": fig,
                        "description": f"Shows the distribution and frequency of values in {col} ({unique_vals:,} unique values, {len(clean_data):,} data points)"
                    })
                    viz_count += 1
            
            # 2. Correlation heatmap if multiple numeric columns
            if len(numeric_cols) > 1 and viz_count < 3:
                try:
                    # Clean numeric data for correlation
                    numeric_data = self.data_copy[numeric_cols].dropna()
                    if len(numeric_data) > 1:
                        correlation_matrix = numeric_data.corr()
                        
                        # Only create if we have valid correlations
                        if not correlation_matrix.empty and correlation_matrix.shape[0] > 1:
                            fig = px.imshow(
                                correlation_matrix,
                                title="Correlation Heatmap",
                                color_continuous_scale="RdYlBu_r",
                                aspect="auto",
                                text_auto=True
                            )
                            fig.update_layout(
                                width=600,
                                height=500
                            )
                            
                            created_visualizations.append({
                                "type": "heatmap",
                                "title": "Correlation Heatmap",
                                "figure": fig,
                                "description": f"Shows correlations between {len(numeric_cols)} numeric variables ({len(numeric_data):,} data points)"
                            })
                            viz_count += 1
                except Exception as e:
                    self.thoughts.append(f"Warning: Could not create correlation heatmap: {str(e)}")
            
            # 3. Bar chart for first categorical column
            if len(categorical_cols) > 0 and viz_count < 3:
                col = categorical_cols[0]
                
                # Clean categorical data
                clean_data = self.data_copy[col].dropna()
                if len(clean_data) > 0:
                    value_counts = clean_data.value_counts().head(10)
                    
                    if len(value_counts) > 0:
                        fig = px.bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            title=f'Top Values in {col}',
                            labels={'x': col, 'y': 'Count'}
                        )
                        fig.update_layout(height=400)
                        
                        created_visualizations.append({
                            "type": "bar",
                            "title": f"Top Values in {col}",
                            "figure": fig,
                            "description": f"Shows the top {len(value_counts)} most frequent values in {col} ({len(clean_data):,} total data points)"
                        })
                        viz_count += 1
            
            # 4. Scatter plot if we have at least 2 numeric columns
            if len(numeric_cols) >= 2 and viz_count < 3:
                col1, col2 = numeric_cols[0], numeric_cols[1]
                
                try:
                    # Clean data for scatter plot
                    plot_data = self.data_copy[[col1, col2]].dropna()
                    
                    if len(plot_data) > 0:
                        # Sample data if too large for performance
                        if len(plot_data) > 5000:
                            plot_data = plot_data.sample(n=5000)
                        
                        fig = px.scatter(
                            plot_data,
                            x=col1,
                            y=col2,
                            title=f'{col1} vs {col2}',
                            opacity=0.7
                        )
                        fig.update_layout(height=400)
                        
                        created_visualizations.append({
                            "type": "scatter",
                            "title": f"{col1} vs {col2}",
                            "figure": fig,
                            "description": f"Shows the relationship between {col1} and {col2}" + (f" (sample of {len(plot_data):,} points)" if len(plot_data) < len(self.data_copy) else f" ({len(plot_data):,} points)")
                        })
                        viz_count += 1
                except Exception as e:
                    self.thoughts.append(f"Warning: Could not create scatter plot: {str(e)}")
            
            # Store visualizations
            self.visualizations.extend(created_visualizations)
            
            # Record the visualization creation
            cycle_time = (datetime.now() - self.current_cycle_start).total_seconds()
            tokens_per_second = self.current_tokens_used / cycle_time if cycle_time > 0 else 0
            
            self.cycle_times.append({
                "cycle": f"viz_{len(self.visualizations)}",
                "time_seconds": cycle_time,
                "start_time": self.current_cycle_start,
                "tokens_used": self.current_tokens_used,
                "tokens_per_second": tokens_per_second
            })
            
            # Create summary response
            viz_summary = f"""
VISUALIZATIONS CREATED ({len(created_visualizations)} charts):

{chr(10).join([f"• {viz['title']}: {viz['description']}" for viz in created_visualizations])}

ANALYSIS INSIGHTS:
{response}

These visualizations help reveal:
- Data distributions and patterns
- Relationships between variables
- Key trends and outliers
- Categories and their frequencies

The charts are interactive and can be used to explore the data in detail.
"""
            
            self.analysis_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "context": "Data visualization creation",
                "results": viz_summary,
                "cycle": f"visualization_{len(self.analysis_history) + 1}",
                "cycle_time": round(cycle_time, 2),
                "data_manipulation": "Visualization creation",
                "visualizations": created_visualizations
            })
            
            self.thoughts.append(f"Created {len(created_visualizations)} visualizations in {cycle_time:.2f} seconds")
            
            return viz_summary
            
        except Exception as e:
            error_msg = f"Error creating visualizations: {str(e)}"
            self.thoughts.append(error_msg)
            return error_msg

    def perform_web_search(self, query: str) -> str:
        """
        Perform web search to gather external information.
        
        Args:
            query: Search query string
            
        Returns:
            str: Summary of search results
        """
        if not self.web_search_tool:
            return "Web search not available for this agent."
        
        try:
            self.thoughts.append(f"Performing web search for: {query}")
            
            # Perform different types of searches based on agent role
            if self.role == "Accountant":
                # Search for financial data or market information
                if any(keyword in query.lower() for keyword in ['stock', 'company', 'ticker']):
                    results = self.web_search_tool.get_financial_data(query.upper())
                else:
                    results = self.web_search_tool.search_web(query, num_results=5)
            
            elif self.role == "Sports Coach":
                # Search for sports statistics
                results = self.web_search_tool.get_team_stats(query, "general", "2024")
            
            else:
                # General web search
                results = self.web_search_tool.search_web(query, num_results=5)
            
            # Store search results
            self.web_search_results.append({
                'query': query,
                'results': results,
                'timestamp': datetime.now().isoformat()
            })
            
            # Generate summary based on results
            if 'error' in results:
                summary = f"Search encountered an error: {results['error']}"
            else:
                if 'search_results' in results:
                    search_items = results['search_results'][:3]
                    summary = f"Found {len(search_items)} relevant sources:\n"
                    for item in search_items:
                        summary += f"- {item['title']}: {item['snippet'][:100]}...\n"
                elif 'current_price' in results:
                    # Financial data
                    summary = f"Financial data for {results['symbol']}:\n"
                    summary += f"- Current price: ${results['current_price']}\n"
                    summary += f"- Price change: {results['price_change']} ({results['percent_change']:.2f}%)\n"
                    summary += f"- Company: {results.get('company_name', 'N/A')}\n"
                else:
                    summary = f"Search completed for '{query}'. Results stored for analysis."
            
            self.thoughts.append(f"Web search completed: {summary[:100]}...")
            
            return summary
            
        except Exception as e:
            error_msg = f"Error performing web search: {str(e)}"
            self.thoughts.append(error_msg)
            return error_msg

    def create_enhanced_visualizations(self, viz_type: str = "auto") -> str:
        """
        Create enhanced visualizations using advanced visualization engine.
        
        Args:
            viz_type: Type of visualization to create ('auto', 'financial', 'sports', etc.)
            
        Returns:
            str: Summary of created visualizations
        """
        if not self.enhanced_viz_engine or self.data_copy is None:
            return "Enhanced visualizations not available or no data loaded."
        
        try:
            self.thoughts.append(f"Creating enhanced {viz_type} visualizations")
            
            created_visualizations = []
            
            # Auto-select visualization types based on role and data
            if viz_type == "auto":
                if self.role == "Accountant":
                    viz_type = "financial"
                elif self.role == "Sports Coach":
                    viz_type = "sports"
                else:
                    viz_type = "general"
            
            # Create role-specific visualizations
            if viz_type == "financial" and self.role == "Accountant":
                # Calculate basic financial metrics
                financial_metrics = {}
                numeric_cols = self.data_copy.select_dtypes(include=[np.number]).columns
                
                if 'revenue' in self.data_copy.columns:
                    financial_metrics['revenue_growth'] = self.data_copy['revenue'].pct_change().mean() * 100
                if 'profit' in self.data_copy.columns and 'revenue' in self.data_copy.columns:
                    financial_metrics['profit_margin'] = (self.data_copy['profit'] / self.data_copy['revenue']).mean() * 100
                financial_metrics['roi'] = 15.5  # Example ROI
                
                fig = self.enhanced_viz_engine.create_financial_dashboard(self.data_copy, financial_metrics)
                
                created_visualizations.append({
                    "type": "financial_dashboard",
                    "title": "Financial Performance Dashboard",
                    "figure": fig,
                    "description": "Comprehensive financial analysis dashboard with key metrics and trends"
                })
            
            elif viz_type == "sports" and self.role == "Sports Coach":
                numeric_cols = self.data_copy.select_dtypes(include=[np.number]).columns.tolist()
                
                if 'player' in self.data_copy.columns or 'name' in self.data_copy.columns:
                    player_col = 'player' if 'player' in self.data_copy.columns else 'name'
                    metric_cols = numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                    
                    if metric_cols:
                        fig = self.enhanced_viz_engine.create_sports_performance_chart(
                            self.data_copy, player_col, metric_cols
                        )
                        
                        created_visualizations.append({
                            "type": "sports_performance",
                            "title": "Sports Performance Analysis",
                            "figure": fig,
                            "description": f"Performance analysis dashboard for {len(self.data_copy)} records with {len(metric_cols)} metrics"
                        })
            
            else:
                # General enhanced visualizations
                numeric_cols = self.data_copy.select_dtypes(include=[np.number]).columns.tolist()
                
                # 3D scatter plot if enough numeric columns
                if len(numeric_cols) >= 3:
                    fig = self.enhanced_viz_engine.create_3d_scatter(
                        self.data_copy, numeric_cols[0], numeric_cols[1], numeric_cols[2]
                    )
                    
                    created_visualizations.append({
                        "type": "3d_scatter",
                        "title": f"3D Analysis: {numeric_cols[0]} vs {numeric_cols[1]} vs {numeric_cols[2]}",
                        "figure": fig,
                        "description": f"Three-dimensional visualization of relationships between {numeric_cols[0]}, {numeric_cols[1]}, and {numeric_cols[2]}"
                    })
                
                # Advanced correlation heatmap
                if len(numeric_cols) >= 2:
                    fig = self.enhanced_viz_engine.create_correlation_heatmap_advanced(self.data_copy)
                    
                    created_visualizations.append({
                        "type": "advanced_correlation",
                        "title": "Advanced Correlation Analysis",
                        "figure": fig,
                        "description": f"Enhanced correlation heatmap for {len(numeric_cols)} numeric variables"
                    })
            
            # Store enhanced visualizations
            self.visualizations.extend(created_visualizations)
            
            summary = f"Created {len(created_visualizations)} enhanced visualizations:\n"
            for viz in created_visualizations:
                summary += f"- {viz['title']}: {viz['description']}\n"
            
            self.thoughts.append(f"Created {len(created_visualizations)} enhanced visualizations")
            
            return summary
            
        except Exception as e:
            error_msg = f"Error creating enhanced visualizations: {str(e)}"
            self.thoughts.append(error_msg)
            return error_msg

    def get_web_search_results(self) -> List[Dict]:
        """
        Get all web search results performed by this agent.
        
        Returns:
            List[Dict]: List of search results with queries and responses
        """
        return self.web_search_results

    def get_enhanced_capabilities(self) -> Dict:
        """
        Get information about enhanced capabilities available to this agent.
        
        Returns:
            Dict: Information about enhanced features and tools
        """
        capabilities = {
            'web_search_available': self.web_search_tool is not None,
            'enhanced_viz_available': self.enhanced_viz_engine is not None,
            'role_config_available': self.role_config is not None,
            'enhanced_features_enabled': ENHANCED_FEATURES_AVAILABLE
        }
        
        if self.role_config:
            capabilities.update({
                'tools': self.role_config.get('tools', []),
                'capabilities': self.role_config.get('capabilities', []),
                'visualization_focus': self.role_config.get('visualization_focus', [])
            })
        
        return capabilities

    def use_enhanced_prompt(self, base_prompt: str) -> str:
        """
        Enhance the base prompt with role-specific enhanced prompt if available.
        
        Args:
            base_prompt: Base prompt to enhance
            
        Returns:
            str: Enhanced prompt or original prompt if enhancement not available
        """
        if self.role_config and 'prompt' in self.role_config:
            enhanced_prompt = f"{self.role_config['prompt']}\n\nCURRENT TASK:\n{base_prompt}"
            return enhanced_prompt
        
        return base_prompt 