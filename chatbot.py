import pathlib
import textwrap
import google.generativeai as genai
import PIL.Image
import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import CORS
import re
import string  # Add this line to use string.punctuation

# Configure API Key
GOOGLE_API_KEY = 'AIzaSyDhEZbtZkV9EN7H6O91M1AMN0r1UXzu5gE'
genai.configure(api_key=GOOGLE_API_KEY)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def format_response(response, max_words=50):  # Increased max_words to 50
    """
    1. Limit response to max_words.
    2. Remove unwanted characters.
    3. Replace multiple spaces/newlines with a single space.
    4. Ensure the result is a clean sentence with proper punctuation.
    """
    # Remove unwanted characters, keep only letters/numbers/spaces and basic punctuation
    cleaned_response = re.sub(r'[^\w\s.,!?]', '', response)

    # Replace multiple spaces/newlines with a single space
    single_line_response = re.sub(r'\s+', ' ', cleaned_response)

    # Strip leading/trailing spaces
    formatted_response = single_line_response.strip()

    # Capitalize the first letter of the response
    if formatted_response:
        formatted_response = formatted_response[0].upper() + formatted_response[1:]

    # Ensure proper punctuation at the end of the response
    if formatted_response and formatted_response[-1] not in string.punctuation:
        formatted_response += "."

    # Limit the response to the specified number of words (max_words)
    words = formatted_response.split()
    truncated_response = ' '.join(words[:max_words])

    return truncated_response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chat_bot():
    """Endpoint to handle chatbot interactions."""
    logger.debug("Request received for chatbot interaction.")

    # Check if the request contains JSON data
    if not request.is_json:
        logger.error("Request must be JSON.")
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    user_input = data.get('message')  # Extract message from the request body

    if not user_input:
        logger.error("No message provided.")
        return jsonify({"error": "No message provided"}), 400

    try:
        # Initialize the GenerativeModel with Gemini
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Modify user_input to include customer care tone for Indian Railways
        user_input = f"Act as a professional customer care assistant for Indian Railways and give response in not more than 30 words. {user_input}"

        # Generate content based on user input
        result = model.generate_content([user_input])
        logger.info("Content generated successfully.")

        return jsonify({"response": format_response(to_markdown(result.text))})

    except Exception as e:
        logger.exception("An error occurred during chatbot processing.")
        return jsonify({"error": str(e)}), 500

def to_markdown(text):
    """Format text to look like Markdown-style.""" 
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

@app.route('/upload', methods=['POST'])
def upload_image():
    """Endpoint to handle image uploads."""
    logger.debug("Request received to upload an image.")
    
    if 'image' not in request.files:
        logger.error("No file part in the request.")
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        logger.error("No selected file.")
        return jsonify({"error": "No selected file"}), 400

    try:
        # Initialize the GenerativeModel with Gemini
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Save the uploaded image temporarily
        img_file_path = f"temp_{file.filename}"
        file.save(img_file_path)
        logger.info(f"Image saved temporarily at {img_file_path}.")
        
        # Open image using PIL
        with PIL.Image.open(img_file_path) as img:
            logger.info("Image opened successfully.")

            # Upload the image to Gemini
            myfile = genai.upload_file(img_file_path)  # Upload the image file

            # Custom prompt to focus on customer care language for Indian Railways
            text_prompt = ("Please analyze this image and describe any issues or abnormalities "
                           "that might be present. Keep it brief and professional, behaving like "
                           "a customer care assistant for Indian Railways. Please say something like: "
                           "'Okay, this is a problem. Let me reroute this and provide you with the quickest help possible. "
                           "I will ensure that this is resolved promptly. Please give a concise response in under 30 words.'")

            # Generate content based on the image and the text prompt
            result = model.generate_content([myfile, text_prompt])  # Send both image and text
            logger.info(f"Analysis result: {result.text}")

        # Clean up the temporary file
        os.remove(img_file_path)
        logger.info(f"Temporary file {img_file_path} removed.")
        return jsonify({"suggestions": format_response(result.text)})

    except Exception as e:
        logger.exception("An error occurred during image processing.")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)  # Set debug=True to enable debug mode
