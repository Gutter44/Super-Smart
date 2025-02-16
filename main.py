from http.server import BaseHTTPRequestHandler
from .crew import CrewAISystem
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

crew_system = CrewAISystem()

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))

        task_name = data.get('task')
        task_params = data.get('params', {})

        if not task_name:
            self.send_error(400, "Missing 'task' in request body")
            return

        try:
            logger.info(f"Executing task: {task_name}")
            result = crew_system.execute_task(task_name, task_params)
            logger.info(f"Task result: {result}")

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            self.send_error(500, f"Internal Server Error: {str(e)}")

    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode('utf-8'))
        else:
            self.send_error(404, "Not Found")

if __name__ == '__main__':
    from http.server import HTTPServer
    port = 8000
    httpd = HTTPServer(('', port), handler)
    print(f"Serving on port {port}")

    # Example usage
    tasks = [
        ("skip_trace", {
            "name": "John Doe",
            "last_known_address": "123 Main St, Anytown, USA",
            "ai_provider": "openai"
        }),
        ("document_processing", {
            "document_url": "https://example.com/document.pdf",
            "document_type": "claim_form",
            "ai_provider": "groq"
        }),
        ("compliance_check", {
            "claim_id": "CLM-12345",
            "jurisdiction": "Oklahoma",
            "ai_provider": "openrouter"
        })
    ]

    for task_name, task_params in tasks:
        result = crew_system.execute_task(task_name, task_params)
        print(f"Task: {task_name}")
        print(f"Result: {result}")
        print("---")

    httpd.serve_forever()

