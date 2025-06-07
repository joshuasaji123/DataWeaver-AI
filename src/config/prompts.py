from typing import Dict

def get_role_prompts() -> Dict:
    """
    Get role-specific prompts and sub-goals.
    
    Returns:
        Dict: Dictionary containing prompts and sub-goals for each role
    """
    return {
        "Business Analyst": {
            "prompt": """As a Business Analyst, analyze the provided data and provide insights focusing on business impact and strategic implications.

Data Summary:
{data_summary}

Context:
{context}

Please provide a detailed analysis that includes:

1. Key Business Metrics:
   - Identify and analyze the most important business metrics in the data
   - Calculate relevant KPIs and their trends
   - Highlight any significant changes or patterns

2. Business Impact Assessment:
   - Evaluate the business implications of the findings
   - Identify potential opportunities and risks
   - Assess the impact on different business units or stakeholders

3. Strategic Recommendations:
   - Provide actionable recommendations based on the data
   - Suggest specific strategies to address identified issues
   - Outline potential next steps for further analysis

4. Competitive Analysis:
   - Compare metrics against industry benchmarks if available
   - Identify competitive advantages or disadvantages
   - Suggest areas for competitive improvement

Please be specific about the data points you're analyzing and provide concrete examples from the data.""",
            "sub_goals": [
                "Identify key business metrics and KPIs",
                "Assess business impact and implications",
                "Provide strategic recommendations",
                "Analyze competitive positioning"
            ]
        },
        "Data Analyst": {
            "prompt": """As a Data Analyst, perform a comprehensive analysis of the provided data focusing on patterns, trends, and data quality.

Data Summary:
{data_summary}

Context:
{context}

Please provide a detailed analysis that includes:

1. Data Quality Assessment:
   - Evaluate data completeness and accuracy
   - Identify any data quality issues or anomalies
   - Suggest data cleaning or preprocessing steps

2. Statistical Analysis:
   - Perform relevant statistical tests
   - Calculate key statistics for each variable
   - Identify correlations and relationships

3. Pattern Recognition:
   - Identify trends and patterns in the data
   - Highlight any unusual or significant observations
   - Analyze seasonal or cyclical patterns if present

4. Visualization Recommendations:
   - Suggest appropriate visualizations for key findings
   - Recommend charts or graphs to highlight patterns
   - Specify which metrics should be tracked over time

Please reference specific data points and provide concrete examples from the dataset.""",
            "sub_goals": [
                "Assess data quality and completeness",
                "Identify patterns and trends",
                "Perform statistical analysis",
                "Recommend visualizations"
            ]
        },
        "Statistician": {
            "prompt": """As a Statistician, conduct a rigorous statistical analysis of the provided data focusing on methodology and statistical significance.

Data Summary:
{data_summary}

Context:
{context}

Please provide a detailed analysis that includes:

1. Statistical Overview:
   - Calculate and interpret key statistical measures
   - Assess the distribution of variables
   - Identify any statistical anomalies

2. Advanced Statistical Analysis:
   - Perform appropriate statistical tests
   - Evaluate statistical significance
   - Analyze relationships between variables

3. Predictive Analysis:
   - Identify potential predictive relationships
   - Suggest appropriate statistical models
   - Assess model assumptions and validity

4. Statistical Recommendations:
   - Recommend additional statistical tests
   - Suggest improvements to the analysis methodology
   - Identify areas requiring further statistical investigation

Please provide specific statistical measures and test results with their interpretations.""",
            "sub_goals": [
                "Perform statistical tests",
                "Analyze relationships",
                "Evaluate significance",
                "Recommend methodology improvements"
            ]
        },
        "Custom": {
            "prompt": """As a {role}, analyze the provided data and provide insights based on your expertise.

Data Summary:
{data_summary}

Context:
{context}

Please provide a detailed analysis that includes:

1. Initial Observations:
   - Key patterns and trends in the data
   - Notable features or characteristics
   - Potential areas of interest

2. Key Findings:
   - Detailed analysis of important aspects
   - Specific insights from the data
   - Relevant examples and evidence

3. Recommendations:
   - Actionable suggestions based on findings
   - Areas for further investigation
   - Potential improvements or solutions

4. Areas for Further Analysis:
   - Additional data points to consider
   - Potential follow-up questions
   - Suggested next steps

Please be specific about the data points you're analyzing and provide concrete examples.""",
            "sub_goals": []
        }
    }

def get_default_prompt_template():
    """
    Get the default prompt template for custom roles.
    Returns the custom role prompt template.
    """
    return get_role_prompts()["Custom"]["prompt"] 