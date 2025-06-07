"""
Enhanced Agent module for the Multi-Agent Data Analysis System with LangChain integration.

This module implements a hybrid approach that combines our sophisticated workflow intelligence
with LangChain's robust agent framework. The EnhancedAgent class inherits from our existing
Agent class while adding LangChain capabilities for:

- Improved LLM communication and error handling
- Advanced tool integration with standardized interfaces  
- Better prompt management with templates and versioning
- Enhanced memory management and conversation handling
- Production-ready monitoring and observability

HYBRID ARCHITECTURE BENEFITS:
- Preserves our intelligent workflow engine (Analyze → Clean → Visualize → Summarize)
- Maintains our three execution modes (Supervised, Unsupervised, Headless)
- Keeps our domain-specific optimizations and data isolation
- Adds LangChain's robustness and ecosystem integration
- Provides future-proofing against AI framework evolution

The hybrid approach ensures we get the best of both worlds: our domain expertise
in data analysis workflows combined with LangChain's infrastructure excellence.
"""

from typing import List, Dict, Optional, Any
import pandas as pd
from datetime import datetime
import time

# LangChain core imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool, BaseTool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage
from langchain_core.callbacks import BaseCallbackHandler

# LangChain model integrations
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

# Import our existing agent infrastructure
from src.models.agent import Agent
from src.config.prompts import get_role_prompts

class EnhancedCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to track token usage and performance metrics."""
    
    def __init__(self, agent_name: str):
        super().__init__()
        self.agent_name = agent_name
        self.token_count = 0
        self.start_time = None
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts running."""
        self.start_time = datetime.now()
        
    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM ends running."""
        if hasattr(response, 'llm_output') and response.llm_output:
            # Extract token usage if available
            if 'token_usage' in response.llm_output:
                self.token_count += response.llm_output['token_usage'].get('total_tokens', 0)

class DataAnalysisToolkit:
    """
    Custom toolkit containing our domain-specific data analysis tools.
    These tools wrap our existing functionality to work with LangChain's tool interface.
    """
    
    def __init__(self, agent: 'EnhancedAgent'):
        self.agent = agent
        self.tools = self._create_tools()
    
    def _create_tools(self) -> List[BaseTool]:
        """Create LangChain tools from our existing agent methods."""
        
        return [
            Tool(
                name="data_analyzer",
                description="""
                Analyze data patterns, statistics, and relationships. Use this tool when you need to:
                - Explore data characteristics and distributions
                - Calculate statistical summaries and metrics
                - Identify patterns, trends, and anomalies
                - Generate insights about data quality and structure
                Input: Data analysis context and specific questions
                """,
                func=self._analyze_data_wrapper
            ),
            Tool(
                name="data_cleaner",
                description="""
                Clean and preprocess data to improve quality. Use this tool when you need to:
                - Handle missing values with intelligent imputation
                - Remove duplicate records and optimize data types
                - Detect and flag outliers using statistical methods
                - Improve data quality for analysis and modeling
                Input: Data cleaning requirements and strategy
                """,
                func=self._clean_data_wrapper
            ),
            Tool(
                name="data_visualizer", 
                description="""
                Create appropriate charts and visualizations based on data characteristics. Use this tool when you need to:
                - Generate histograms, scatter plots, correlation heatmaps
                - Create bar charts for categorical data analysis
                - Build interactive visualizations with proper chart selection
                - Visualize patterns and relationships in the data
                Input: Visualization requirements and data context
                """,
                func=self._visualize_data_wrapper
            ),
            Tool(
                name="workflow_analyzer",
                description="""
                Analyze current workflow state and make intelligent decisions about next actions. Use this tool when you need to:
                - Determine optimal next step in analysis workflow
                - Assess analysis completeness and depth
                - Evaluate data quality needs and cleaning requirements
                - Make autonomous decisions about task continuation
                Input: Current workflow context and progress status
                """,
                func=self._analyze_workflow_wrapper
            )
        ]
    
    def _analyze_data_wrapper(self, context: str) -> str:
        """Wrapper for our data analysis functionality."""
        try:
            if self.agent.data_copy is not None:
                # Use our existing sophisticated analysis logic
                result = self.agent.analyze(
                    self.agent.data_copy, 
                    context=f"LangChain enhanced analysis: {context}"
                )
                return f"Analysis completed: {result}"
            else:
                return "Error: No data available for analysis"
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def _clean_data_wrapper(self, strategy: str) -> str:
        """Wrapper for our data cleaning functionality."""
        try:
            # Use our existing intelligent data cleaning
            result = self.agent.manipulate_data()
            return f"Data cleaning completed: {result}"
        except Exception as e:
            return f"Cleaning error: {str(e)}"
    
    def _visualize_data_wrapper(self, requirements: str) -> str:
        """Wrapper for our visualization functionality."""
        try:
            # Use our existing intelligent visualization creation
            result = self.agent.create_visualizations()
            return f"Visualizations created: {result}"
        except Exception as e:
            return f"Visualization error: {str(e)}"
    
    def _analyze_workflow_wrapper(self, context: str) -> str:
        """Wrapper for our workflow analysis functionality."""
        try:
            # Use our sophisticated workflow decision logic
            next_action = self.agent.choose_action(context)
            
            # Provide detailed reasoning about the decision
            workflow_state = {
                "analyses_completed": len([h for h in self.agent.analysis_history if h.get('data_manipulation') != 'Visualization creation']),
                "cleanings_performed": len(self.agent.data_manipulations),
                "visualizations_created": len(self.agent.visualizations),
                "summaries_generated": len([h for h in self.agent.analysis_history if str(h.get('cycle', '')).startswith('summary')])
            }
            
            return f"""
            Workflow Analysis Results:
            - Recommended next action: {next_action}
            - Current progress: {workflow_state}
            - Reasoning: Based on our intelligent workflow engine analysis
            """
        except Exception as e:
            return f"Workflow analysis error: {str(e)}"

class EnhancedAgent(Agent):
    """
    Enhanced Agent class that integrates LangChain capabilities with our existing workflow intelligence.
    
    This hybrid approach preserves all our competitive advantages while adding LangChain's robustness:
    - Keeps our intelligent workflow decision engine
    - Maintains our three execution modes (Supervised, Unsupervised, Headless)
    - Preserves our data isolation and performance monitoring
    - Adds LangChain's improved error handling and tool integration
    - Provides access to LangChain's ecosystem and future innovations
    """
    
    def __init__(self, name: str, role: str, model: str = "deepseek-coder:latest", use_openai: bool = False):
        """
        Initialize enhanced agent with LangChain integration.
        
        Args:
            name (str): Agent name
            role (str): Agent role (Business Analyst, Data Scientist, etc.)
            model (str): Model to use for analysis
            use_openai (bool): Whether to use OpenAI API
        """
        # Initialize parent class with all existing functionality
        super().__init__(name, role, model, use_openai)
        
        # Initialize LangChain components
        self.llm = self._initialize_llm()
        self.callback_handler = EnhancedCallbackHandler(self.name)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        # Create our domain-specific toolkit
        self.toolkit = DataAnalysisToolkit(self)
        
        # Initialize LangChain agent with our tools
        self.agent_executor = self._create_agent_executor()
        
        # Track LangChain enhancements
        self.langchain_enabled = True
        self.enhanced_responses = []
        
    def _initialize_llm(self):
        """Initialize the appropriate LangChain LLM based on configuration."""
        try:
            if self.use_openai:
                return ChatOpenAI(
                    model=self.model,
                    temperature=0.1,
                    callbacks=[self.callback_handler],
                    request_timeout=60
                )
            else:
                return Ollama(
                    model=self.model,
                    callbacks=[self.callback_handler],
                    temperature=0.1
                )
        except Exception as e:
            self.thoughts.append(f"LLM initialization error: {str(e)}")
            # Fallback to parent class behavior
            return None
    
    def _create_agent_executor(self) -> Optional[AgentExecutor]:
        """Create LangChain agent executor with our tools and prompts."""
        try:
            if not self.llm:
                return None
                
            # Create enhanced prompt template that incorporates our role-based logic
            role_info = get_role_prompts().get(self.role, get_role_prompts()["Custom"])
            
            prompt_template = PromptTemplate.from_template("""
            You are {role}, an expert in data analysis with the following specialized capabilities:
            
            Role Description: {role_description}
            Goals: {goals}
            
            You have access to the following tools for data analysis:
            {tools}
            
            IMPORTANT: You must follow the intelligent workflow: Analyze → Clean → Visualize → Summarize → Repeat/End
            
            Use the following format:
            Question: the input question or task
            Thought: think about what to do based on the workflow state
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question
            
            Current Context: {input}
            Chat History: {chat_history}
            Agent Scratchpad: {agent_scratchpad}
            """)
            
            # Format the prompt with role-specific information
            formatted_prompt = prompt_template.partial(
                role=self.role,
                role_description=role_info["prompt"],
                goals=", ".join(role_info.get("sub_goals", []))
            )
            
            # Create ReAct agent with our tools
            agent = create_react_agent(
                llm=self.llm,
                tools=self.toolkit.tools,
                prompt=formatted_prompt
            )
            
            return AgentExecutor(
                agent=agent,
                tools=self.toolkit.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5,
                max_execution_time=300  # 5 minute timeout
            )
            
        except Exception as e:
            self.thoughts.append(f"Agent executor creation error: {str(e)}")
            return None
    
    def _generate_response(self, prompt: str) -> str:
        """
        Enhanced response generation using LangChain with fallback to parent implementation.
        
        This method first tries to use LangChain's robust agent framework, and falls back
        to our original implementation if needed. This ensures reliability while adding
        LangChain's advanced capabilities.
        """
        try:
            if self.agent_executor and self.langchain_enabled:
                # Use LangChain agent executor for enhanced response generation
                start_time = datetime.now()
                
                response = self.agent_executor.invoke({
                    "input": prompt,
                    "chat_history": self.memory.chat_memory.messages if self.memory else []
                })
                
                # Track performance
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # Update token tracking from callback handler
                self.current_tokens_used = self.callback_handler.token_count
                
                # Store enhanced response information
                self.enhanced_responses.append({
                    "timestamp": start_time,
                    "duration": duration,
                    "tokens_used": self.current_tokens_used,
                    "response_length": len(response.get("output", "")),
                    "success": True
                })
                
                self.thoughts.append(f"✅ LangChain enhanced response generated in {duration:.2f}s")
                
                return response.get("output", "No response generated")
                
            else:
                # Fallback to parent class implementation
                self.thoughts.append("🔄 Using fallback response generation")
                return super()._generate_response(prompt)
                
        except Exception as e:
            # Log error and fallback to parent implementation
            self.thoughts.append(f"⚠️ LangChain error: {str(e)}, using fallback")
            
            # Track failed response
            self.enhanced_responses.append({
                "timestamp": datetime.now(),
                "error": str(e),
                "success": False
            })
            
            # Disable LangChain temporarily if multiple failures
            if len([r for r in self.enhanced_responses[-5:] if not r.get("success", True)]) >= 3:
                self.langchain_enabled = False
                self.thoughts.append("🛡️ Temporarily disabled LangChain due to multiple failures")
            
            return super()._generate_response(prompt)
    
    def choose_action(self, context: str = "") -> str:
        """
        Enhanced action selection that combines our workflow intelligence with LangChain reasoning.
        
        This method preserves our sophisticated workflow decision logic while optionally
        enhancing it with LangChain's reasoning capabilities for complex scenarios.
        """
        try:
            # First, use our existing intelligent workflow logic
            base_decision = super().choose_action(context)
            
            # For complex decisions, enhance with LangChain reasoning
            if (self.agent_executor and self.langchain_enabled and 
                base_decision in ["analyze", "summarize"] and 
                len(self.analysis_history) > 2):
                
                enhanced_context = f"""
                Current workflow decision: {base_decision}
                Context: {context}
                Progress: {len(self.analysis_history)} analyses, {len(self.data_manipulations)} cleanings, 
                {len(self.visualizations)} visualizations completed
                
                As a {self.role}, provide additional insights about this decision or suggest optimizations.
                """
                
                try:
                    enhanced_reasoning = self.agent_executor.invoke({
                        "input": enhanced_context
                    })
                    
                    reasoning = enhanced_reasoning.get("output", "")
                    self.thoughts.append(f"🧠 LangChain enhanced reasoning: {reasoning[:200]}...")
                    
                    # For now, we keep our base decision but log the enhanced reasoning
                    # Future iterations could incorporate this into decision-making
                    
                except Exception as e:
                    self.thoughts.append(f"⚠️ Enhanced reasoning failed: {str(e)}")
            
            return base_decision
            
        except Exception as e:
            self.thoughts.append(f"⚠️ Action selection error: {str(e)}")
            return super().choose_action(context)
    
    def get_langchain_stats(self) -> Dict:
        """Get statistics about LangChain usage and performance."""
        if not self.enhanced_responses:
            return {"langchain_enabled": self.langchain_enabled, "total_enhanced_responses": 0}
        
        successful_responses = [r for r in self.enhanced_responses if r.get("success", False)]
        
        return {
            "langchain_enabled": self.langchain_enabled,
            "total_enhanced_responses": len(self.enhanced_responses),
            "successful_responses": len(successful_responses),
            "success_rate": len(successful_responses) / len(self.enhanced_responses) * 100,
            "avg_response_time": sum(r.get("duration", 0) for r in successful_responses) / max(len(successful_responses), 1),
            "total_tokens_used": sum(r.get("tokens_used", 0) for r in successful_responses),
            "tools_available": len(self.toolkit.tools) if hasattr(self, 'toolkit') else 0
        }
    
    def toggle_langchain(self, enabled: bool = None) -> bool:
        """Toggle LangChain integration on/off for testing or fallback scenarios."""
        if enabled is not None:
            self.langchain_enabled = enabled
        else:
            self.langchain_enabled = not self.langchain_enabled
            
        self.thoughts.append(f"🔄 LangChain integration {'enabled' if self.langchain_enabled else 'disabled'}")
        return self.langchain_enabled

def create_enhanced_agent(name: str, role: str, model: str = "deepseek-coder:latest", use_openai: bool = False) -> EnhancedAgent:
    """
    Factory function to create enhanced agents with LangChain integration.
    
    This function provides a clean interface for creating enhanced agents while
    handling any initialization errors gracefully.
    """
    try:
        agent = EnhancedAgent(name, role, model, use_openai)
        agent.thoughts.append(f"✅ Enhanced agent created with LangChain integration")
        return agent
    except Exception as e:
        # Fallback to basic agent if enhancement fails
        from src.models.agent import Agent
        basic_agent = Agent(name, role, model, use_openai)
        basic_agent.thoughts.append(f"⚠️ Created basic agent due to enhancement error: {str(e)}")
        return basic_agent 