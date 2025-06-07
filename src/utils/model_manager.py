import psutil
import subprocess
from typing import List, Dict
import ollama

class ModelManager:
    """
    Manages model-related operations including:
    - System resource monitoring
    - Model availability checking
    - Model size parsing
    - Model recommendations
    """
    
    @staticmethod
    def get_system_resources() -> Dict:
        """Get system resources information."""
        memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
        cpu_count = psutil.cpu_count()
        return {
            "memory_gb": round(memory_gb, 2),
            "cpu_count": cpu_count
        }

    @staticmethod
    def parse_model_size(size_str: str) -> float:
        """Convert model size string to gigabytes."""
        try:
            if 'GB' in size_str:
                return float(size_str.replace('GB', '').strip())
            elif 'MB' in size_str:
                return float(size_str.replace('MB', '').strip()) / 1024
            return 0.0
        except:
            return 0.0

    @staticmethod
    def get_available_models() -> List[Dict]:
        """Get list of available Ollama models with their sizes."""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    models = []
                    for line in lines[1:]:
                        parts = line.split()
                        if len(parts) >= 3:
                            name = parts[0]
                            size = parts[2]
                            models.append({
                                'name': name,
                                'size_gb': ModelManager.parse_model_size(size)
                            })
                    return models
            
            try:
                models = ollama.list()
                if isinstance(models, dict) and 'models' in models:
                    return [{'name': model['name'], 'size_gb': 0.0} for model in models['models']]
            except Exception:
                pass
            
            return []
        except Exception as e:
            raise Exception(f"Error fetching models: {str(e)}")

    @staticmethod
    def get_model_requirements() -> Dict:
        """Get model requirements and recommendations."""
        return {
            "deepseek-r1:32b": {
                "min_memory_gb": 16,
                "recommended_memory_gb": 32,
                "description": "Advanced model for complex analysis",
                "size_gb": 19.0
            },
            "mixtral:latest": {
                "min_memory_gb": 16,
                "recommended_memory_gb": 32,
                "description": "High-performance model for general analysis",
                "size_gb": 26.0
            },
            "qwen2.5-coder:32b": {
                "min_memory_gb": 16,
                "recommended_memory_gb": 32,
                "description": "Specialized for code and technical analysis",
                "size_gb": 19.0
            },
            "GandalfBaum/deepseek_r1-claude3.7:latest": {
                "min_memory_gb": 8,
                "recommended_memory_gb": 16,
                "description": "Balanced model for general analysis",
                "size_gb": 9.0
            },
            "nomic-embed-text:latest": {
                "min_memory_gb": 4,
                "recommended_memory_gb": 8,
                "description": "Lightweight model for basic analysis",
                "size_gb": 0.274
            },
            "codellama:34b": {
                "min_memory_gb": 16,
                "recommended_memory_gb": 32,
                "description": "Specialized for code analysis",
                "size_gb": 19.0
            },
            "deepseek-coder:latest": {
                "min_memory_gb": 8,
                "recommended_memory_gb": 16,
                "description": "Efficient model for code analysis",
                "size_gb": 0.776
            }
        }

    @staticmethod
    def get_model_recommendations(role: str, goals: List[str]) -> Dict:
        """Get model recommendations based on role and goals."""
        recommendations = {
            "ollama_models": [],
            "openai_models": []
        }
        
        # Role-specific recommendations
        if role == "Business Analyst":
            recommendations["ollama_models"] = [
                {
                    "name": "mixtral:latest",
                    "reason": "Excellent for business analysis with strong reasoning capabilities",
                    "priority": "high"
                },
                {
                    "name": "deepseek-r1:32b",
                    "reason": "Strong analytical capabilities for business insights",
                    "priority": "medium"
                }
            ]
            recommendations["openai_models"] = [
                {
                    "name": "gpt-4-turbo-preview",
                    "reason": "Best for complex business analysis and strategic thinking",
                    "priority": "high"
                },
                {
                    "name": "gpt-3.5-turbo",
                    "reason": "Good for basic business analysis with faster response times",
                    "priority": "medium"
                }
            ]
        elif role == "Data Analyst":
            recommendations["ollama_models"] = [
                {
                    "name": "codellama:34b",
                    "reason": "Strong data processing and pattern recognition capabilities",
                    "priority": "high"
                },
                {
                    "name": "qwen2.5-coder:32b",
                    "reason": "Excellent for technical data analysis and visualization",
                    "priority": "medium"
                }
            ]
            recommendations["openai_models"] = [
                {
                    "name": "gpt-4-turbo-preview",
                    "reason": "Best for complex data analysis and pattern recognition",
                    "priority": "high"
                },
                {
                    "name": "gpt-3.5-turbo",
                    "reason": "Good for basic data analysis tasks",
                    "priority": "medium"
                }
            ]
        elif role == "Statistician":
            recommendations["ollama_models"] = [
                {
                    "name": "deepseek-r1:32b",
                    "reason": "Strong mathematical and statistical capabilities",
                    "priority": "high"
                },
                {
                    "name": "mixtral:latest",
                    "reason": "Good for statistical analysis and hypothesis testing",
                    "priority": "medium"
                }
            ]
            recommendations["openai_models"] = [
                {
                    "name": "gpt-4-turbo-preview",
                    "reason": "Best for complex statistical analysis and mathematical reasoning",
                    "priority": "high"
                },
                {
                    "name": "gpt-3.5-turbo",
                    "reason": "Good for basic statistical analysis",
                    "priority": "medium"
                }
            ]
        
        # Add goal-specific recommendations
        for goal in goals:
            if "market" in goal.lower() or "competitive" in goal.lower():
                recommendations["ollama_models"].append({
                    "name": "mixtral:latest",
                    "reason": "Strong capabilities in market analysis and competitive intelligence",
                    "priority": "high"
                })
            elif "visualization" in goal.lower():
                recommendations["ollama_models"].append({
                    "name": "qwen2.5-coder:32b",
                    "reason": "Excellent for data visualization and presentation",
                    "priority": "high"
                })
            elif "hypothesis" in goal.lower() or "statistical" in goal.lower():
                recommendations["ollama_models"].append({
                    "name": "deepseek-r1:32b",
                    "reason": "Strong statistical analysis capabilities",
                    "priority": "high"
                })
        
        # Remove duplicates and sort by priority
        for model_type in recommendations:
            seen = set()
            recommendations[model_type] = [
                model for model in recommendations[model_type]
                if not (model["name"] in seen or seen.add(model["name"]))
            ]
            recommendations[model_type].sort(key=lambda x: x["priority"] == "high", reverse=True)
        
        return recommendations 