from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import replicate
import os
import json
import time
from dotenv import load_dotenv
#adding uvicorn for loading
import uvicorn
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
# from starlette.responses import FileResponse

app = FastAPI()

class CourseInfo(BaseModel):
    vak: str
    onderwerp: str
    duur: str
    ects: int
    voorkennis: str
    taal: str
    api_key: str

# @app.get("/")
# # def read_root():
# #     return {"message": "Welcome to the EMA API"}

# Mount the static directory to serve static files@
app.mount("/static", StaticFiles(directory="static"), name="static")

#show the html in the index.html file at /
@app.get("/")
def read_root():
    return FileResponse('static/index.html')


def get_system_prompt():
    return ("Your Role: As an experienced and excellent university lecturer, you deliver high-quality educational material extremely quickly, "
    "based on recent scientific insights and relevant literature, with references according to APA guidelines. You develop dynamic and inclusive educational material. "
    "Note: You aim to impress my colleagues and demonstrate that you can develop educational material very quickly and to a very high standard, "
    "perhaps even better than they could themselves. You are a subject matter expert in the relevant fields and also a top educator in terms of teaching methods. "
    "You strictly adhere to your assignments and execute them meticulously, thoroughly, excellently, and faster than anyone else."
    "\n\n"
    "Your Task: Deliver high-quality educational materials at lightning speed, aligned with the latest trends and research in the field. "
    "Whenever possible, incorporate the use of generative AI in the materials you generate, so that students learn to work with it (free versions only). "
    "Additionally, always include literature references to the (scientific) information you used to generate the material."
    "\n\n"
    "You deliver various products, such as lessons, case studies, assignments, study guides, multiple-choice tests, and educational programs. "
    "Each product must be current, challenging, educational, and aligned with the curriculum. The data provided by the user should be incorporated "
    "into all the material you deliver, and you should not deviate from it, except for the literature used (for which you may make suggestions). "
    "When proposing literature, it should always be real (preferably scientific) literature, referenced according to APA guidelines. "
    "An educational program typically spans 8 weeks and includes introductions, weekly themes, preparatory activities, and recent literature. "
    "One ECTS represents 28 hours of work, which can be self-study, lectures, tutorials, or other activities. "
    "A study guide includes an introduction with the module's goal, importance, and relevance, supported by arguments. It also provides a complete and detailed overview of course information, learning objectives, "
    "educational and learning activities, study materials, assessment, planning, rules, contact information, and weekly assignments. "
    "Students may use generative AI, provided they properly cite its use according to APA guidelines, ensure its use is ethically responsible, and still meet the learning objectives. "
    "Provide clear frameworks and guidelines for this in a separate section of the study guide. Ensure all information is well-structured and immediately usable, maintaining a friendly tone. "
    "Educational programs should be fun, varied, well-structured, and appealing to the target audience, with assignments relevant to the program and practical applications in the relevant professional field. "
    "If a user asks you to create something unfamiliar to you, ask the minimum necessary questions to gain clarity before starting the creation process."
    "\n\n")

def read_course_data(file_path):
    # Read values from the JSON file
    with open(file_path) as f:
        data = json.load(f)
        # Select the course in the JSON file
        data_course = data['course']
        for key, value in data_course.items():
            print(f"{key.capitalize()}: {value}")

    return data_course

def create_prompt_lesson_plan(course):
    prompt = f"""Give me an extensive well written lesson plan for the course {course.vak}, the module 
           takes {course.duur} and the course consists of {course.ects} ECTS. Students have the 
           folllowing prior/foreknowledge knowledge: {course.voorkennis}. Finally present it in {course.taal}. 
           It is very important that the whole answer is translated in {course.taal}."""
    return prompt

def create_prompt_lesson(course):
    prompt = f"""As an expert in {course.vak}, create a 90-minute lesson for HBO students on {course.onderwerp} with clear learning objectives:

    1. Introduction (10 mins): Introduce the topic, its relevance, goals, and learning outcomes.
    2. Core (60 mins):
    - Theory (20 mins): Explain key concepts with examples.
    - Activities (20 mins): Practice exercises, case studies, discussions.
    - Application (20 mins): Practical application with industry examples.
    3. Conclusion (10 mins): Summarize key points, answer questions, assign follow-up tasks.

    Instructions:
    - Use clear language suitable for HBO students.
    - Ensure interactivity with questions and discussions.
    - Provide practical examples and tips.

    Design the lesson according to this structure and present it in {course.taal}."""
    return prompt

def create_prompt_assignment(course):
    prompt = f"""As an expert in {course.vak}, create an HBO-level assignment on {course.onderwerp} that is challenging and relevant. 
        The assignment should include:
            1. Introduction:
            - Overview of the topic and its importance.
            - Objectives and expected learning outcomes.
            2. Tasks:
            - Theoretical questions and practical exercises.
            - Application of key concepts and theories.
            - Real-world scenarios or case studies.
            3. Instructions:
            - Step-by-step guidelines.
            - Required deliverables and submission format.
            - Deadlines.
            4. Evaluation:
            - Grading rubric and criteria.
            - Weight of each task/component.
            5. Resources:
            - Recommended readings and tools.
            - References to relevant literature.

            **Guidelines**:
            - Appropriate for HBO-level students.
            - Clear and understandable language.
            - Opportunities for knowledge and practical skills demonstration.
            - Encourage critical thinking and problem-solving.

            Design the assignment according to these guidelines and present it in {course.taal}."""

    return prompt

def create_prompt_formative_exam(course):
    prompt = f"""As an expert in {course.vak}, create formative exam questions for HBO students on the topic of {course.onderwerp}. 
        Very important! Every question should be translated in {course.taal}
        The questions should be designed to assess understanding and provide feedback for improvement. Ensure the exam includes the following elements:
            1. Question Types:
            - A mix of multiple-choice, short answer, and essay questions.
            - Questions that cover key concepts, theories, and practical applications.
            2. Question Content:
            - Clear and concise questions relevant to the subject matter.
            - Include real-world scenarios or case studies.
            - Questions that encourage critical thinking and problem-solving.
            3. Instructions:
            - Provide clear instructions for each question type.
            - Specify the expected length and depth of responses where applicable.
            4. Feedback :
            - Suggest one or ore model answers or key points to cover in responses.

            **Specific Guidelines**:
            - Present it in {course.taal} language
            - Ensure the questions are appropriate for HBO-level students.
            - Use clear and understandable language.
            - Cover a range of difficulty levels to differentiate student understanding.
            - Encourage application of knowledge to practical situations.

            Design the exam questions according to these guidelines. Very important! The whole text and every question should be shown in {course.taal}
            """
    return prompt


def streamed_reply_llama3(prompt, system_prompt):
    all_events = ""
    # The meta/meta-llama-3-70b-instruct model can stream output as it's running.
    start_time = time.time()
    for event in replicate.stream(
        "meta/meta-llama-3-70b-instruct",
        input={
            "top_k": 50,
            "top_p": 0.9,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_tokens": 4000,
            "min_tokens": 0,
            "temperature": 0.1,
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "presence_penalty": 1.15,
            "frequency_penalty": 0.2
        },
    ):
        print(str(event), end="")
        all_events += str(event)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time} seconds")

    # Convert all_events string (completion) to a dictionary
    output = {"completion": all_events}
    return output

@app.post("/generate-lesson-plan/")
def generate_lesson_plan(course: CourseInfo):
    os.environ['REPLICATE_API_TOKEN'] = course.api_key
    system_prompt = get_system_prompt()
    prompt = create_prompt_lesson_plan(course)
    print("Prompt:", prompt)
    response_json = streamed_reply_llama3(prompt, system_prompt)
    return response_json

@app.post("/generate-lesson/")
def generate_lesson(course: CourseInfo):
    os.environ['REPLICATE_API_TOKEN'] = course.api_key
    system_prompt = get_system_prompt()
    prompt = create_prompt_lesson(course)
    print("Prompt:", prompt)
    response_json = streamed_reply_llama3(prompt, system_prompt)
    return response_json

@app.post("/generate-assignment/")
def generate_assignment(course: CourseInfo):
    os.environ['REPLICATE_API_TOKEN'] = course.api_key
    system_prompt = get_system_prompt()
    prompt = create_prompt_assignment(course)
    print("Prompt:", prompt)
    response_json = streamed_reply_llama3(prompt, system_prompt)
    return response_json

@app.post("/generate-exam/")
def generate_exam(course: CourseInfo):
    os.environ['REPLICATE_API_TOKEN'] = course.api_key
    system_prompt = get_system_prompt()
    prompt = create_prompt_formative_exam(course)
    print("Prompt:", prompt)
    response_json = streamed_reply_llama3(prompt, system_prompt)
    return response_json


# For local testing outside of the FastAPI
if __name__ == "__main__":
    # load_dotenv()
    # generate_exam(CourseInfo(vak="Computer Science", onderwerp="Artificial Intelligence", duur="8 weeks", ects=5, voorkennis="Python programming", taal="English", api_key=os.environ['REPLICATE_API_TOKEN']))
    uvicorn.run(app, host="0.0.0.0", port=80)