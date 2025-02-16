import os
import pandas as pd
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from typing import List, Optional

# Initialize the LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model_name='llama3-8b-8192'
)

# Define Agents
problem_definer = Agent(
    role='Problem Definer',
    goal='Define and clarify the machine learning problem',
    backstory='Expert in translating business needs into ML problems',
    verbose=True,
    allow_delegation=True,
    llm=llm
)

data_assessor = Agent(
    role='Data Assessor',
    goal='Assess the quality and suitability of the data for the ML problem',
    backstory='Specialist in data analysis and preprocessing',
    verbose=True,
    allow_delegation=True,
    llm=llm
)

model_recommender = Agent(
    role='Model Recommender',
    goal='Recommend suitable ML models and approaches',
    backstory='Expert in various ML algorithms and their applications',
    verbose=True,
    allow_delegation=True,
    llm=llm
)

code_generator = Agent(
    role='Code Generator',
    goal='Generate sample code for the recommended ML approach',
    backstory='Experienced ML engineer proficient in Python and ML libraries',
    verbose=True,
    allow_delegation=True,
    llm=llm
)

# Define Tasks
def define_problem_task(user_input: str) -> Task:
    return Task(
        description=f"Define the ML problem based on user input: {user_input}",
        agent=problem_definer
    )

def assess_data_task(file_path: str) -> Task:
    return Task(
        description=f"Assess the data in the file: {file_path}",
        agent=data_assessor
    )

def recommend_model_task() -> Task:
    return Task(
        description="Recommend suitable ML models based on the problem and data assessment",
        agent=model_recommender
    )

def generate_code_task() -> Task:
    return Task(
        description="Generate sample Python code for the recommended ML approach",
        agent=code_generator
    )

def load_csv_file(file_path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded CSV file with {len(df)} rows and {len(df.columns)} columns.")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        print("Proceeding without data assessment.")
        return None

def create_tasks(user_input: str, file_path: Optional[str]) -> List[Task]:
    tasks = [define_problem_task(user_input)]
    
    if file_path:
        df = load_csv_file(file_path)
        if df is not None:
            tasks.append(assess_data_task(file_path))
    
    tasks.extend([recommend_model_task(), generate_code_task()])
    return tasks

def save_results(user_input: str, result: str) -> None:
    with open('ml_assistant_output.md', 'w') as f:
        f.write("# CrewAI Machine Learning Assistant Output\n\n")
        f.write(f"## User Input\n{user_input}\n\n")
        f.write(f"## Analysis Results\n{result}\n")
    
    print("\nResults have been saved to ml_assistant_output.md")

def main():
    print("Welcome to the CrewAI Machine Learning Assistant!")
    user_input = input("Please describe your machine learning problem: ")
    
    file_path = input("Enter the path to your CSV file (or press Enter to skip): ").strip()
    
    tasks = create_tasks(user_input, file_path)
    
    crew = Crew(
        agents=[problem_definer, data_assessor, model_recommender, code_generator],
        tasks=tasks,
        verbose=True
    )
    
    result = crew.kickoff()
    
    save_results(user_input, result)

if __name__ == "__main__":
    main()

