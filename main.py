import logging
from typing import Dict
from .crew import CrewAISystem

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        crew_system = CrewAISystem()
        
        # Example task executions
        tasks = [
            ("skip_trace", {
                "name": "John Doe",
                "last_known_address": "123 Main St, Anytown, USA"
            }),
            ("document_processing", {
                "document_url": "https://example.com/document.pdf",
                "document_type": "claim_form"
            }),
            ("compliance_check", {
                "claim_id": "CLM-12345",
                "jurisdiction": "Oklahoma"
            })
        ]

        for task_name, task_params in tasks:
            logger.info(f"Executing task: {task_name}")
            result = crew_system.execute_task(task_name, task_params)
            logger.info(f"Task result: {result}")

        # Demonstrate agent capabilities
        agents = ["compliance_agent", "paralegal_agent"]
        for agent in agents:
            capabilities = crew_system.get_agent_capabilities(agent)
            logger.info(f"{agent} capabilities: {capabilities}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

