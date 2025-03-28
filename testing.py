import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY= os.getenv('GEMINI_API_KEY')  

genai.configure(api_key=GEMINI_API_KEY)

generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 512,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-002",
    generation_config=generation_config,
    system_instruction=(
        "You need to play the role of a doctor. The patients will share symptoms to you and you need to prescribe OTC drugs with dosage to them to just address the symptoms. "
        "Along with this, you are to recommend precautions to be taken and the diet to be followed. The response format is as follows:\n"
        "<drug/syrup/ointment prescribed>: x-y-z [where x, y, z are three courses of the day morning, evening, and night along with a little extra information on timing 1 if true 0 if not true eg:1-0-0 = dosage in morning]\n"
        "<recommended diet>\n"
        "<precaution>\n"
        "<report summary> [in not more than 40 words]\n"
        "The entire response should be very brief and strictly stick to the format."
    ),
)

def query_gemini(prompt):
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)
    return response.text

