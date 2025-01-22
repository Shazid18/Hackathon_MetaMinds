import os
from flask import Flask, jsonify, render_template, request
from crewai import LLM, Agent, Task, Crew
from crewai_tools import SerperDevTool
import warnings

# Initialize Flask app
app = Flask(__name__)

# Setup for warnings and tools
warnings.filterwarnings('ignore')

# Initialize LLM and other tools
llm = LLM(
    model="gpt-4o",
    api_key=""
)

os.environ["SERPER_API_KEY"] = ""

search_tool = SerperDevTool()

# Define agents
weather_agent = Agent(
    role='Weather Analyst',
    goal='Accurately predict and analyze weather conditions for the specified location',
    backstory="""You are an experienced meteorologist with expertise in weather analysis and forecasting. Your job is to provide accurate weather information to help travelers plan their trips safely.""",
    tools=[search_tool],
    verbose=True,
    llm=llm
)

safety_agent = Agent(
    role='Safety Advisor',
    goal='Provide safety precautions based on weather conditions',
    backstory="""You are a travel safety expert with deep knowledge of how weather conditions affect travel safety. You provide crucial safety advice to ensure travelers are well-prepared.""",
    tools=[search_tool],
    verbose=True,
    llm=llm
)

tour_planner = Agent(
    role='Tour Planner',
    goal='Create optimal tour plans considering weather conditions',
    backstory="""You are an experienced tour planner who specializes in creating adaptable travel itineraries based on weather conditions and location-specific attractions.""",
    tools=[search_tool],
    verbose=True,
    llm=llm
)

medical_advisor = Agent(
    role='Medical Advisor',
    goal='Identify potential medical risks and provide preventive advice',
    backstory="""You are a travel medicine specialist who helps travelers understand and prepare for potential medical issues they might face during their journey.""",
    tools=[search_tool],
    verbose=True,
    llm=llm
)

emergency_locator = Agent(
    role='Emergency Services Locator',
    goal='Provide information about local emergency services and medical facilities',
    backstory="""You are a local emergency services expert who maintains updated information about medical facilities and emergency contacts in various locations.""",
    tools=[search_tool],
    verbose=True,
    llm=llm
)

insurance_advisor = Agent(
    role='Insurance Advisor',
    goal='Assess travel insurance needs and provide insurance recommendations',
    backstory="""You are an experienced insurance advisor specializing in travel and health insurance. Your expertise helps travelers make informed decisions about their insurance needs based on their destination, duration of stay, and existing coverage.""",
    tools=[search_tool],
    verbose=True,
    llm=llm
)

supervisor_agent = Agent(
    role='Travel Advisory Supervisor',
    goal='Compile and organize all travel advisory information into a comprehensive report',
    backstory="""You are a senior travel advisor who specializes in creating comprehensive travel reports. You analyze and organize information from various travel experts to create clear, well-structured travel advisory reports.""",
    tools=[search_tool],
    verbose=True,
    llm=llm
)

def serialize_crew_output(crew_output):
    """
    Convert CrewOutput object to a JSON-serializable format
    """
    if hasattr(crew_output, 'raw_output'):
        return str(crew_output.raw_output)
    return str(crew_output)

# Function to create tasks
def create_task(description, agent, expected_output, context=None):
    return Task(
        description=description,
        agent=agent,
        expected_output=expected_output,
        context=context if context else []
    )

# Function to process initial tasks
def process_initial_tasks(location):
    weather_task = create_task(
        f"Analyze and predict the current and upcoming weather conditions for {location}. Include temperature, precipitation, and any weather warnings.",
        weather_agent,
        "Detailed weather analysis and forecasts for the specified location"
    )
    
    safety_task = create_task(
        f"Based on the weather analysis for {location}, provide detailed safety precautions and recommendations for travelers. Include what to pack and what to avoid.",
        safety_agent,
        "Comprehensive safety recommendations based on weather conditions",
        [weather_task]
    )
    
    tour_task = create_task(
        f"Create a flexible tour plan for {location} considering the weather conditions. Include indoor and outdoor activities with alternatives for bad weather.",
        tour_planner,
        "Detailed tour itinerary with weather-based alternatives",
        [weather_task]
    )
    
    medical_task = create_task(
        f"Identify potential medical conditions and health risks travelers might face in {location}. Provide preventive measures and recommendations.",
        medical_advisor,
        "Complete medical risk assessment and preventive measures",
        [weather_task]
    )
    
    emergency_task = create_task(
        f"""Compile a list of emergency services in {location}, including:
        - Hospitals and their specialties
        - 24/7 pharmacies
        - Emergency contact numbers
        - Medical facilities' addresses""",
        emergency_locator,
        "Comprehensive list of emergency services and contacts",
        []
    )
    
    # Create crew and run tasks
    crew = Crew(
        agents=[weather_agent, safety_agent, tour_planner, medical_advisor, emergency_locator],
        tasks=[weather_task, safety_task, tour_task, medical_task, emergency_task]
    )
    
    results = crew.kickoff(inputs={"location": location})
    return serialize_crew_output(results)

# Function to process insurance task
def process_insurance_task(location, previous_results):
    insurance_task = create_task(
        f"""Based on the completed analysis for {location}, provide detailed insurance information including:
        - Key benefits of travel insurance for {location}
        - List of insurance providers offering travel insurance in {location}
        - Coverage details, contact information, and websites for each provider""",
        insurance_advisor,
        "Insurance information with providers, coverage, and benefits",
        []
    )
    
    insurance_crew = Crew(
        agents=[insurance_advisor],
        tasks=[insurance_task]
    )
    
    insurance_result = insurance_crew.kickoff(inputs={
        "location": location,
        "previous_results": previous_results
    })
    
    return serialize_crew_output(insurance_result)

# Function to compile final report
def compile_final_report(location, initial_results, insurance_results):
    supervisor_task = create_task(
        f"""Create a comprehensive travel advisory report for {location} by combining and organizing:
        1. Initial analysis results: {initial_results}
        2. Insurance recommendations: {insurance_results}
        
        Format the report with clear sections for:
        - Weather Analysis
        - Safety Precautions
        - Tour Planning
        - Medical Risks
        - Emergency Services
        - Insurance Recommendations
        
        Format the output as a proper Markdown document with headers, subheaders, and appropriate formatting.""",
        supervisor_agent,
        "Complete travel advisory report with all sections organized in Markdown format",
        []
    )
    
    supervisor_crew = Crew(
        agents=[supervisor_agent],
        tasks=[supervisor_task]
    )
    
    final_report = supervisor_crew.kickoff(inputs={
        "location": location,
        "initial_results": initial_results,
        "insurance_results": insurance_results
    })
    
    return serialize_crew_output(final_report)

# Function to save report to file
def save_report_to_file(location, report_content):
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"reports/travel_advisory_{location.lower().replace(' ', '_')}_{timestamp}.md"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(str(report_content))
    
    return filename

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_travel_advisory', methods=['POST'])
def get_travel_advisory():
    try:
        location = request.form['location']
        
        # Process tasks
        initial_results = process_initial_tasks(location)
        insurance_results = process_insurance_task(location, initial_results)
        final_report = compile_final_report(location, initial_results, insurance_results)
        
        # Save report
        saved_file = save_report_to_file(location, final_report)
        
        # Return JSON response
        return jsonify({
            'status': 'success',
            'final_report': final_report,
            'saved_file': saved_file
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)