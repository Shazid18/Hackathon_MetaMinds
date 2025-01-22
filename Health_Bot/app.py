from flask import Flask, render_template, request
import os
import warnings
from crewai import LLM, Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# Your existing setup for agents, tools, and tasks
llm = LLM(
    model="gpt-4o",
    api_key="YOUR_API_KEY"
)
os.environ["SERPER_API_KEY"] = "ebf2218969d785f46bed48d0b2bc0cd4232b8642"

search_tool = SerperDevTool()

# Agents and task setups (same as before)
weather_agent = Agent(
    role='Weather Analyst',
    goal='Accurately predict and analyze weather conditions for the specified location',
    backstory="""You are an experienced meteorologist with expertise in weather analysis 
    and forecasting. Your job is to provide accurate weather information to help travelers 
    plan their trips safely.""",
    tools=[search_tool],
    verbose=True,
    llm=llm
)

safety_agent = Agent(
    role='Safety Advisor',
    goal='Provide safety precautions based on weather conditions',
    backstory="""You are a travel safety expert with deep knowledge of how weather 
    conditions affect travel safety. You provide crucial safety advice to ensure 
    travelers are well-prepared.""",
    tools=[search_tool],
    verbose=True,
    llm=llm
)

tour_planner = Agent(
    role='Tour Planner',
    goal='Create optimal tour plans considering weather conditions',
    backstory="""You are an experienced tour planner who specializes in creating 
    adaptable travel itineraries based on weather conditions and location-specific 
    attractions.""",
    tools=[search_tool],
    verbose=True,
    llm=llm
)

medical_advisor = Agent(
    role='Medical Advisor',
    goal='Identify potential medical risks and provide preventive advice',
    backstory="""You are a travel medicine specialist who helps travelers understand 
    and prepare for potential medical issues they might face during their journey.""",
    tools=[search_tool],
    verbose=True,
    llm=llm
)

emergency_locator = Agent(
    role='Emergency Services Locator',
    goal='Provide information about local emergency services and medical facilities',
    backstory="""You are a local emergency services expert who maintains updated 
    information about medical facilities and emergency contacts in various locations.""",
    tools=[search_tool],
    verbose=True,
    llm=llm
)

# Insurance Advisor Agent
insurance_advisor = Agent(
    role='Insurance Advisor',
    goal='Assess travel insurance needs and provide insurance recommendations',
    backstory="""You are an experienced insurance advisor specializing in travel and health 
    insurance. Your expertise helps travelers make informed decisions about their insurance 
    needs based on their destination, duration of stay, and existing coverage.""",
    tools=[search_tool],
    verbose=True,
    llm=llm
)

# Tasks setup (same as before)
weather_task = Task(
    description="Analyze and predict the current and upcoming weather conditions for {location}. Include temperature, precipitation, and any weather warnings.",
    agent=weather_agent,
    expected_output="The predicted weather conditions for the location, including temperature, precipitation, and warnings.",
    output_file="outputs/weather_analysis.md",
    create_directory=True
)

safety_task = Task(
    description="Based on the weather analysis for {location}, provide detailed safety precautions and recommendations for travelers. Include what to pack and what to avoid.",
    agent=safety_agent,
    expected_output="Detailed safety precautions and recommendations based on the weather conditions.",
    context=[weather_task],
    output_file="outputs/safety_precautions.md",
    create_directory=True
)

planning_task = Task(
    description="Create a flexible tour plan for {location} considering the weather conditions. Include indoor and outdoor activities with alternatives for bad weather.",
    agent=tour_planner,
    expected_output="A flexible tour plan with both indoor and outdoor activities.",
    context=[weather_task],
    output_file="outputs/tour_plan.md",
    create_directory=True
)

medical_task = Task(
    description="Identify potential medical conditions and health risks travelers might face in {location}. Provide preventive measures and recommendations.",
    agent=medical_advisor,
    expected_output="List of potential medical risks and preventive measures for travelers.",
    context=[weather_task],
    output_file="outputs/medical_risks.md",
    create_directory=True
)

emergency_task = Task(
    description="""Compile a list of emergency services in {location}, including:
    - Hospitals and their specialties
    - 24/7 pharmacies
    - Emergency contact numbers
    - Medical facilities' addresses""",
    agent=emergency_locator,
    expected_output="List of emergency services, including hospitals, pharmacies, and contact details.",
    output_file="outputs/emergency_services.md",
    create_directory=True
)

def process_initial_tasks(location):
    # Create initial crew without insurance agent
    initial_crew = Crew(
        agents=[weather_agent, safety_agent, tour_planner,
                medical_advisor, emergency_locator],
        tasks=[weather_task, safety_task, planning_task,
               medical_task, emergency_task]
    )
    
    # Run initial tasks
    initial_result = initial_crew.kickoff(inputs={"location": location})
    return initial_result

def process_insurance_task(location, has_insurance, previous_results):
    # Insurance task that runs after user input
    insurance_task = Task(
        description=f"""Based on the completed analysis for {location} and user's insurance status ({has_insurance}):
        1. If user has no insurance:
           - Research and recommend suitable travel health insurance options
           - List coverage benefits and costs
           - Provide application process details
        2. If user has existing insurance:
           - Analyze coverage gaps for {location}
           - Recommend supplementary insurance if needed
           - Provide tips for using existing insurance abroad
        
        Consider the medical risks and emergency services already identified in the previous analysis.""",
        agent=insurance_advisor,
        expected_output="Personalized insurance recommendations and guidance based on user's current coverage and destination.",
        context=[medical_task, emergency_task],
        output_file="outputs/insurance_recommendations.md",
        create_directory=True
    )
    
    # Create insurance crew
    insurance_crew = Crew(
        agents=[insurance_advisor],
        tasks=[insurance_task]
    )
    
    # Run insurance task
    insurance_result = insurance_crew.kickoff(inputs={
        "location": location,
        "has_insurance": has_insurance,
        "previous_results": previous_results
    })
    return insurance_result

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        location = request.form['location']
        
        # Process initial tasks
        initial_results = process_initial_tasks(location)
        
        # Now ask about insurance
        has_insurance = request.form['has_insurance'].lower() == 'yes'
        
        # Process insurance task
        insurance_results = process_insurance_task(location, has_insurance, initial_results)
        
        # Combine results to show
        complete_results = f"{initial_results}\n\nInsurance Analysis:\n{insurance_results}"
        
        # Return results in a structured way to be displayed in the HTML
        return render_template('index.html', results=complete_results, location=location)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
