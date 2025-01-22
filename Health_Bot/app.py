import os
from flask import Flask, render_template, request
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

# Define agents (same as your previous code)
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
    return results

# Function to process insurance task
def process_insurance_task(location, has_insurance, previous_results):
    insurance_task = create_task(
        f"""Based on the completed analysis for {location} and user's insurance status ({has_insurance}):
        1. If user has no insurance:
           - Research and recommend suitable travel health insurance options
           - List coverage benefits and costs
           - Provide application process details
        2. If user has existing insurance:
           - Analyze coverage gaps for {location}
           - Recommend supplementary insurance if needed
           - Provide tips for using existing insurance abroad""",
        insurance_advisor,
        "Detailed insurance recommendations and guidance",
        []
    )
    
    insurance_crew = Crew(
        agents=[insurance_advisor],
        tasks=[insurance_task]
    )
    
    insurance_result = insurance_crew.kickoff(inputs={
        "location": location,
        "has_insurance": has_insurance,
        "previous_results": previous_results
    })
    
    return insurance_result

# Function to compile the final report
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
        
        Format the output as a proper Markdown document with headers, subheaders, and appropriate formatting.
        Use proper Markdown syntax for:
        - Headers (# for main headers, ## for subheaders)
        - Bullet points (- for lists)
        - Emphasis (* for italic, ** for bold)
        - Tables (if needed)
        
        Highlight key warnings and recommendations in each section using bullet points.
        For each section, provide key points in the form of bullet points, not paragraphs.
        """,
        supervisor_agent,
        "Complete travel advisory report with all sections organized in Markdown format using bullet points",
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
    
    return final_report

# Function to save the report to a file
def save_report_to_file(location, report_content):
    # Create a reports directory if it doesn't exist
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    # Format filename with location and current timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"reports/travel_advisory_{location.lower().replace(' ', '_')}_{timestamp}.md"
    
    # Save the report
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(str(report_content))
    
    return filename

# Flask route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to handle form submission and return results
@app.route('/get_travel_advisory', methods=['POST'])
def get_travel_advisory():
    location = request.form['location']
    has_insurance = request.form['has_insurance']
    
    # Process initial tasks
    initial_results = process_initial_tasks(location)
    
    # Process insurance task
    insurance_results = process_insurance_task(location, has_insurance, initial_results)
    
    # Generate final report
    final_report = compile_final_report(location, initial_results, insurance_results)
    
    # Save the report to a file
    saved_file = save_report_to_file(location, final_report)
    
    return render_template('index.html', location=location, final_report=final_report, saved_file=saved_file)

if __name__ == "__main__":
    app.run(debug=True)
