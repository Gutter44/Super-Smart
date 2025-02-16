import os
import time
import logging
from typing import Dict, List
import yaml
from dotenv import load_dotenv
from .tools.custom_tool import APITool
from .ai_providers.ai_providers import OpenAIProvider, GroqProvider, OpenRouterProvider

class CrewAISystem:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CrewAISystem, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        load_dotenv()
        self.load_configurations()
        self.initialize_tools()
        self.initialize_ai_providers()
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def load_configurations(self):
        try:
            with open("src/my_project/config/agents.yaml", "r") as f:
                self.agent_config = yaml.safe_load(f)
            with open("src/my_project/config/tasks.yaml", "r") as f:
                self.task_config = yaml.safe_load(f)
        except FileNotFoundError as e:
            self.logger.error(f"Configuration file not found: {e}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration: {e}")
            raise

    def initialize_tools(self):
        self.tools = {
            "true_people_search": APITool(os.getenv("TRUE_PEOPLE_SEARCH_API_KEY")),
            "bland_ai": APITool(os.getenv("BLAND_AI_API_KEY")),
            "perplexity": APITool(os.getenv("PERPLEXITY_API_KEY")),
            "make": APITool(os.getenv("MAKE_API_KEY")),
            "google_gemini": APITool(os.getenv("GOOGLE_GEMINI_API_KEY")),
            "endato": APITool(os.getenv("ENDATO_API_KEY")),
        }

    def initialize_ai_providers(self):
        self.ai_providers = {
            "openai": OpenAIProvider(),
            "groq": GroqProvider(),
            "openrouter": OpenRouterProvider(),
        }

    def execute_task(self, task_name: str, params: Dict) -> Dict:
        task_config = self.task_config.get(task_name)
        if not task_config:
            self.logger.error(f"Task {task_name} not found")
            return {"error": f"Task {task_name} not found"}

        max_attempts = task_config["retry_config"]["max_attempts"]
        delay_seconds = task_config["retry_config"]["delay_seconds"]

        for attempt in range(max_attempts):
            try:
                self.logger.info(f"Executing task {task_name} (attempt {attempt + 1}/{max_attempts})")
                result = self._perform_task(task_name, params)
                self.logger.info(f"Task {task_name} completed successfully")
                return {"status": "success", "task": task_name, "result": result}
            except Exception as e:
                self.logger.warning(f"Task {task_name} failed: {str(e)}")
                if attempt < max_attempts - 1:
                    self.logger.info(f"Retrying in {delay_seconds} seconds...")
                    time.sleep(delay_seconds)
                else:
                    self.logger.error(f"Task {task_name} failed after {max_attempts} attempts")
                    return {"status": "error", "task": task_name, "error": str(e)}

    def _perform_task(self, task_name: str, params: Dict) -> Dict:
        required_apis = self.task_config[task_name]["required_apis"]
        for api in required_apis:
            if api.lower() not in self.tools and api.lower() not in self.ai_providers:
                raise ValueError(f"Required API {api} not initialized")
        
        # Example of using an AI provider
        ai_provider = params.get("ai_provider", "openai")
        if ai_provider not in self.ai_providers:
            raise ValueError(f"AI provider {ai_provider} not available")
        
        prompt = params.get("prompt", f"Perform task: {task_name}")
        ai_response = self.ai_providers[ai_provider].generate_text(prompt)
        
        # Simulating additional task execution
        time.sleep(1)  # Simulate some work
        return {"task_output": f"AI Response: {ai_response}", "task_name": task_name}

    def get_agent_capabilities(self, agent_name: str) -> List[str]:
        agent = self.agent_config.get(agent_name)
        if not agent:
            self.logger.warning(f"Agent {agent_name} not found")
            return []
        return agent.get("capabilities", [])

