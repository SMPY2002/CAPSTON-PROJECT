from flask import Flask, render_template, request, jsonify, redirect, send_file, url_for
import requests
from werkzeug.utils import secure_filename
import numpy as np
import pickle
import time
import os
from math import ceil
from collections import Counter
from PIL import Image
import numpy as np
import tensorflow as tf
from fpdf import FPDF
import textwrap
import google.generativeai as genai
import re
from IPython.display import display
from IPython.display import Markdown
import PIL.Image
from dotenv import load_dotenv


# Load the TensorFlow Lite model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def to_markdown(text):
  text = text.replace('*',' •')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


from nltk.corpus import stopwords
# Define English stopwords
stop_words = set(stopwords.words('english'))

app = Flask(__name__)


# Load the trained model
interpreter = load_tflite_model('skin_cancer_model.tflite')
model_2 = pickle.load(open("Thyroid_Dectection.pkl", "rb"))
# Define the mapping of class indices to labels and descriptions
classes = {
    4: ('nv', 'melanocytic nevi'),
    6: ('mel', 'melanoma'),
    2: ('bkl', 'benign keratosis-like lesions'),
    1: ('bcc', 'basal cell carcinoma'),
    5: ('vasc', 'pyogenic granulomas and hemorrhage'),
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
    3: ('df', 'dermatofibroma')
}

# GPT - API KEY Configuration
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the Gemini model
model_1 = genai.GenerativeModel('gemini-pro')
model_3 = genai.GenerativeModel('gemini-pro-vision')
# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

#  Serve static files during development
app.static_folder = 'static'

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/blog_details')
def blog_detail():
    return render_template('blog_details.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/test')
def test():
    return render_template('test_form.html')

@app.route('/loader_animation')
def loader_animation():
    return render_template('loader_animation.html')

@app.route('/cancer_test')
def cancer_test():
    return render_template('cancer_test.html')

@app.route('/thyroid')
def thyroid():
    return render_template('thyroid.html')

@app.route('/heart_attack')
def heart_attack():
    return render_template('heart_disease.html')

# <--------------------------- Heart attack Model ------------------------------------->

@app.route('/submit_heatdisease', methods=['POST'])
def submit_heatdisease():
    heart_data = {
        "name": request.form.get("name"),
        "age": int(request.form.get("age")) if request.form.get("age") else 0,
    }

    print(heart_data)
    
    # Check if the post request has the file part
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return 'No selected file'

    if file:
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            img = PIL.Image.open(f"{file_path}")
            # Perform further processing here
            ecg_response = model_3.generate_content(["Generate a report for whose ECG image is provided, and Tell that what percent of chance that their may be possibility heart attack or any Serious Heart Disease.", img])
            ecg_response.resolve()
            return render_template('heart_disease.html', ecg_response=ecg_response.text, show_test_result=True)
        except Exception as e:
            return f'An error occurred: {str(e)}'


# <--------------------------- Thyroid Model ------------------------------------->    
@app.route("/submit_thyroid", methods=["POST"])
def predict_thyroid():
    thyroid_data = {
        "age": int(request.form.get("age")),
        "tt4_value": int(request.form.get("tt4_value")) if request.form.get("tt4_value") else 0,
        "t3_value": int(request.form.get("t3_value")) if request.form.get("t3_value") else 0,
        "t4u_value": int(request.form.get("t4u_value")) if request.form.get("t4u_value") else 0,
        "fti_value": int(request.form.get("fti_value")) if request.form.get("fti_value") else 0,
        "tsh_value": int(request.form.get("tsh_value")) if request.form.get("tsh_value") else 0,
        "pregnency": 0, # No pregnancy in the form data
    }

    thyroid_data_array = np.array(
        [
            [
                thyroid_data["age"],
                thyroid_data["tt4_value"],
                thyroid_data["t3_value"],
                thyroid_data["t4u_value"],
                thyroid_data["fti_value"],
                thyroid_data["tsh_value"],
                thyroid_data["pregnency"],
            ]
        ]
    )
    time.sleep(5)
    pred_thyroid = model_2.predict(thyroid_data_array)
    if pred_thyroid == 0:
        return render_template("thyroid.html", pred_thyroid="Normal" , show_test_result = True)
    elif pred_thyroid == 1:
        return render_template("thyroid.html", pred_thyroid="Hypo-Thyroid", show_test_result = True)
    else:
        return render_template("thyroid.html", pred_thyroid="Hyper-Thyroid", show_test_result = True)
    
# <--------------------------- Thyroid Model End ------------------------------------->

#  <--------------------------- Get data from Take a Test Form and process it (starts here) ------------------------------>
# Function to calculate BMI
def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

@app.route("/submit", methods=["POST"])
def predict():
    data = {
        "name": request.form.get("name"),
        "age": int(request.form.get("age")) if request.form.get("age") else 0,
        "gender": request.form.get("gender"),
        "height": int(request.form.get("height")) if request.form.get("height") else 0,
        "weight": int(request.form.get("weight")) if request.form.get("weight") else 0,
        "any_existing_disease": [],
        "is_alcoholic": request.form.get("alcohol"),
        "exercise_data" : [],
        "diet": request.form.getlist("diet"),
        "fast_food_consumption": int(request.form.get("fastFoodConsumption")) if request.form.get("fastFoodConsumption") else 0,
        "sleep_hours": int(request.form.get("sleepHours")) if request.form.get("sleepHours") else 0,
        "others": request.form.get("other_info")
    }

    # Initialize any_existing_disease as an empty list
    data["any_existing_disease"] = []

    # Retrieve diseases and their values
    no_checkbox_selected = False  # Flag to track if "NO" checkbox is selected

    for key, value in request.form.items():
        if key.startswith("diseaseValue"):
            disease_name = key.split('-')[1]  # Extract disease name from input name
            disease_value = value
            data["any_existing_disease"].append({disease_name: disease_value})
        elif value == "NO":
            no_checkbox_selected = True

    # If "NO" checkbox is selected, store the value as 0
    if no_checkbox_selected:
        data["any_existing_disease"] = 0

    # Retrieve exercise types and their durations

    # Check if exercise option is selected
    exercise_option = request.form.get("exerciseOption")
    if exercise_option == "yes":
        exercise_duration = int(request.form.get("exerciseDuration")) if request.form.get("exerciseDuration") else 0
        data["exercise_data"].append({"exercise_duration": exercise_duration})

    # Now exercise_data contains a list of dictionaries, each representing an exercise type and its duration
    # Calculate BMI
    bmi = calculate_bmi(data['weight'], data['height'])
    if data['exercise_data']:
        exercise_duration = data['exercise_data'][0]['exercise_duration']
    else:
        exercise_duration = 0
    # Generate medical report
    medical_report_response = model_1.generate_content(f"""Medical Report for {data['name']}:

    Name: {data['name']}
    Age: {data['age']}
    Gender: {data['gender']}
    Height: {data['height']} cm
    Weight: {data['weight']} kg
    BMI: {bmi:.2f}

    Existing Conditions:
    {', '.join([f"- {list(disease.keys())[0]}: {list(disease.values())[0]}" for disease in data['any_existing_disease']]) if data['any_existing_disease'] else "None"}

    Alcohol Consumption: {data['is_alcoholic']}
    Exercise Duration: {data['exercise_data'][0]['exercise_duration']} minutes/day
    Diet: {', '.join(data['diet'])}
    Fast Food Consumption: {data['fast_food_consumption']} times/month
    Sleep Hours: {data['sleep_hours']} hours/day

    Provide detail information to  {data['name']} , including its health plus points and its bad points., and also provide the solution to improve the health of {data['name']}.
    Health Condition and Recommendations:
    [Insert health condition assessment and recommendations based on user's health parameters]

    Daily Exercise Chart:
    [Insert personalized exercise chart based on user's health parameters]

    Food Chart:
    [Insert personalized food chart based on user's health parameters]

    Articles and Resources:
    [Insert links to articles and resources related to improving health]

    Provide health score based on his provided  information. Score should be between 0-10 where 0 is  the lowest and 10 is the highest possible
    """)

    # medical_report = medical_report_response.text
    # Remove headers using various heading symbol combinations
    text_report_general = re.sub(r'^#{1,6} .*', '', medical_report_response.text, flags=re.MULTILINE)

    # Remove bold text wrapped in asterisks or underscores
    text_report_general = re.sub(r'\*{2,}(.*?)\*{2,}|_{2,}(.*?)_{2,}', r'\1', text_report_general)

    # Remove italic text wrapped in single asterisks or underscores
    text_report_general = re.sub(r'\*(.*?)\*|_(.*?)_', r'\1', text_report_general)

    # Remove strikethrough using two tildes
    text_report_general = re.sub(r'~{2,}(.*?)~{2,}', r'\1', text_report_general)

    # Convert code blocks to plain text with indentation
    text_report_general = re.sub(r'`(.*?)`', r'\n`\n\1\n`\n', text_report_general, flags=re.DOTALL)
    text_report_general = text_report_general.replace('```', '\t')

    # Remove inline code wrapped in backticks
    text_report_general = re.sub(r'`(.*?)`', r'\1', text_report_general)

    # Remove quoted blocks using '>' symbol
    text_report_general = re.sub(r'(^|\n)> [^\n]*\n', '\n', text_report_general, flags=re.MULTILINE)

    # Remove horizontal lines
    text_report_general = re.sub(r'^[=\-]+\n', '', text_report_general, flags=re.MULTILINE)

    # Convert lists to plain text with indentation
    text_report_general = re.sub(r'\n(^\s*[-+*] )+(.+?)\n', r'\n\t\1\2\n', text_report_general, flags=re.MULTILINE)

    # Remove extra line breaks
    text_report_general = re.sub(r'\n{2,}', '\n\n', text_report_general)
    final_report = text_report_general.strip()
    # Redirect to the show_result route with medical report data as string
    if final_report:
        return redirect(url_for('show_result', medical_report=final_report))
    else:
        return "Error: Medical report could not be generated."

@app.route("/show_result")
def show_result():
    medical_report = request.args.get('medical_report')

    if medical_report:
        return render_template("result.html", medical_report=medical_report)
    else:
        return "Error: Medical report data is missing."


def make_prediction(interpreter, image_path):
     # Load and preprocess the new image
    new_image = Image.open(image_path)
    new_image = new_image.resize((28, 28))  # Resize the image to match the input size of the model
    new_image = np.array(new_image) / 255.0  # Normalize the image (0-1 range)

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Reshape and set the input tensor
    input_data = np.expand_dims(new_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get predictions
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(predictions)

    # Get the label and description for the predicted class
    predicted_label, predicted_description = classes[predicted_class]

    return predicted_description


# Function to generate detailed description using ChatGPT
def generate_detailed_description(predicted_description,prompt="Provide a detailed description of this disease :"):
    model_2 = genai.GenerativeModel('gemini-pro')

    response = model_2.generate_content(f"{prompt} {predicted_description}")
    # Convert Markdown to HTML
    # Remove headers using various heading symbol combinations
    text = re.sub(r'^#{1,6} .*', '', response.text, flags=re.MULTILINE)

    # Remove bold text wrapped in asterisks or underscores
    text = re.sub(r'\*{2,}(.*?)\*{2,}|_{2,}(.*?)_{2,}', r'\1', text)

    # Remove italic text wrapped in single asterisks or underscores
    text = re.sub(r'\*(.*?)\*|_(.*?)_', r'\1', text)

    # Remove strikethrough using two tildes
    text = re.sub(r'~{2,}(.*?)~{2,}', r'\1', text)

    # Convert code blocks to plain text with indentation
    text = re.sub(r'`(.*?)`', r'\n`\n\1\n`\n', text, flags=re.DOTALL)
    text = text.replace('```', '\t')

    # Remove inline code wrapped in backticks
    text = re.sub(r'`(.*?)`', r'\1', text)

    # Remove quoted blocks using '>' symbol
    text = re.sub(r'(^|\n)> [^\n]*\n', '\n', text, flags=re.MULTILINE)

    # Remove horizontal lines
    text = re.sub(r'^[=\-]+\n', '', text, flags=re.MULTILINE)

    # Convert lists to plain text with indentation
    text = re.sub(r'\n(^\s*[-+*] )+(.+?)\n', r'\n\t\1\2\n', text, flags=re.MULTILINE)

    # Remove extra line breaks
    text = re.sub(r'\n{2,}', '\n\n', text)

    return text.strip()

def generate_pdf_report(predicted_description, detailed_description, filename):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(0, 0, 255)
    pdf.cell(0, 10, "Medical Report", ln=True, align="C")
    pdf.ln(10)

    # Predicted Description Section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Predicted Disease:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(0)  # Reset text color to black
    pdf.multi_cell(0, 10, predicted_description)
    pdf.ln(5)

    # Detailed Description Section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Detailed Description:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, detailed_description)
    pdf.ln(10)

    # Signature Section with Cursive Style
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, "Signature:", ln=True, align="R")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, "MediScan360", ln=True, align="R")
    pdf.ln(10)


    report_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf.output(report_path)


# Uploading image file from the Take a test form and save it to upload folder
@app.route('/upload', methods=['POST'])
def upload():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return 'No selected file'

    if file:
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            predicted_description = make_prediction(interpreter,file_path)
            detailed_description = generate_detailed_description(predicted_description, prompt="Generate a detailed report for the disease:")

            generate_pdf_report(predicted_description, detailed_description, 'report.pdf')

            return f'<a href="/download/{filename}">Download Report</a>'
        except Exception as e:
            return f'An error occurred: {str(e)}'

        
# Route for downloading report
@app.route('/download/<filename>')
def download(filename):
    report_path = os.path.join(app.config['UPLOAD_FOLDER'], 'report.pdf')
    return send_file(report_path, as_attachment=True)

    
#  <--------------------------- Get data from Take a Test Form and process it (ends here) ------------------------------>


# <---------------------- blog.html page function (starts here)  ------------------------->

# Function to fetch health-related blog data from News API
def get_health_blogs():
    
    # Fetch health-related news from around the world
    world_news_url = f'https://newsapi.org/v2/everything?q=health&apiKey={NEWS_API_KEY}'
    world_news_response = requests.get(world_news_url)
    if world_news_response.status_code == 200:
        world_news_data = world_news_response.json()
        world_news = world_news_data.get('articles', [])
    else:
        print(f"Failed to fetch world health news. Status code: {world_news_response.status_code}")
        world_news = []
    
    # Fetch health-related schemes and laws related to India
    india_schemes_laws_url = f'https://newsapi.org/v2/everything?q=health+India&apiKey={NEWS_API_KEY}'
    india_schemes_laws_response = requests.get(india_schemes_laws_url)
    if india_schemes_laws_response.status_code == 200:
        india_schemes_laws_data = india_schemes_laws_response.json()
        india_schemes_laws = india_schemes_laws_data.get('articles', [])
    else:
        print(f"Failed to fetch health schemes and laws related to India. Status code: {india_schemes_laws_response.status_code}")
        india_schemes_laws = []

    # Process health-related blogs
    health_blogs = []
    for article in world_news:
        if all(field in article for field in ['title', 'description', 'url', 'author', 'publishedAt', 'urlToImage']):
            if article['title'] and article['description'] and article['url'] and article['author'] and article['publishedAt'] and article['urlToImage']:
                # Remove time part from publishedAt
                published_at = article['publishedAt'].split('T')[0] 
                health_blog = {
                    'title': article['title'],
                    'description': article['description'],
                    'url': article['url'],
                    'author': article['author'],
                    'publishedAt': published_at,
                    'image': article['urlToImage']
                }
                health_blogs.append(health_blog)

    return health_blogs, india_schemes_laws, world_news

# Stopwords for English language
stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
    'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

# Blog route

@app.route('/blog')
def health_blog():
    query = request.args.get('query')
    if query:
        # Search query is provided, fetch search results
        url = f'https://newsapi.org/v2/everything/india+health?q={query}&apiKey={NEWS_API_KEY}'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            health_blogs = []
            tags = []  # List to store tags
            for article in data.get('articles', [])[:50]:  # Limit to 50 articles
                if all(field in article for field in ['title', 'description', 'url', 'author', 'publishedAt']):
                    if article['title'] and article['description'] and article['url'] and article['author'] and article['urlToImage'] and article['publishedAt']:
                        # Remove time part from publishedAt
                        published_at = article['publishedAt'].split('T')[0]
                        health_blog = {
                            'title': article['title'],
                            'description': article['description'],
                            'url': article['url'],
                            'author': article['author'],
                            'image': article['urlToImage'],
                            'publishedAt': published_at
                        }
                        health_blogs.append(health_blog)
                    
                        # Extract tags from title and description, excluding stopwords
                        title_words = [word.lower() for word in article['title'].split() if word.lower() not in stop_words]
                        description_words = [word.lower() for word in article['description'].split() if word.lower() not in stop_words]
                        tags.extend(title_words)
                        tags.extend(description_words)
                    
            # Count occurrences of each tag
            tag_counts = Counter(tags)
            
            # Select top 10 most common tags
            top_tags = tag_counts.most_common(10)
            
            # Pagination logic
            page = request.args.get('page', 1, type=int)
            per_page = 5  # Number of blog posts per page
            total_pages = ceil(len(health_blogs) / per_page)
            start_index = (page - 1) * per_page
            end_index = min(start_index + per_page, len(health_blogs))
            paginated_blogs = health_blogs[start_index:end_index]
            
            return render_template('blog.html', blog_posts=paginated_blogs, tags=top_tags, pagination={'total_pages': total_pages, 'current_page': page})
        else:
            print(f"API request failed with status code: {response.status_code}")
            return render_template('blog.html', error_message="Failed to fetch search results. Please try again later.")
    else:
        # No search query, display recent health blogs
        blog_posts, india_schemes_laws, _ = get_health_blogs()
        recent_posts = blog_posts[:4] + india_schemes_laws[:1] # Displaying 5 recent posts
        
        tags = []  # List to store tags
        for post in blog_posts:
            # Extract tags from title and description, excluding stopwords
            title_words = [word.lower() for word in post['title'].split() if word.lower() not in stop_words]
            description_words = [word.lower() for word in post['description'].split() if word.lower() not in stop_words]
            tags.extend(title_words)
            tags.extend(description_words)
        
        # Count occurrences of each tag
        tag_counts = Counter(tags)
        
        # Select top 10 most common tags
        top_tags = tag_counts.most_common(10)
        
        # Pagination logic
        page = request.args.get('page', 1, type=int)
        per_page = 5  # Number of blog posts per page
        total_pages = ceil(len(blog_posts) / per_page)
        start_index = (page - 1) * per_page
        end_index = min(start_index + per_page, len(blog_posts))
        paginated_main_posts = blog_posts[start_index:end_index] + india_schemes_laws[start_index:end_index]
        
        # Fetch Instagram feed using images from health blogs
        instagram_feed = []
        for post in blog_posts[:6]:  # Displaying 6 images
            instagram_post = {
                'image': post['image'],
                'url': post['url']
            }
            instagram_feed.append(instagram_post)
        
        return render_template('blog.html', blog_posts=paginated_main_posts, recent_posts=recent_posts, instagram_feed=instagram_feed, tags=top_tags, pagination={'total_pages': total_pages, 'current_page': page})

# Tag cloud 
@app.route('/tag_search/<tag>')
def tag_search(tag):
    # Search for blog posts containing the tag
    url = f'https://newsapi.org/v2/everything?q={tag}&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        health_blogs = []
        tags = []  # List to store tags
        for article in data.get('articles', [])[:10]:  # Limit to 10 articles
            if all(field in article for field in ['title', 'description', 'url', 'author', 'publishedAt', 'urlToImage']):
                if article['title'] and article['description'] and article['url'] and article['author'] and article['publishedAt'] and article['urlToImage']:
                        # Remove time part from publishedAt
                        published_at = article['publishedAt'].split('T')[0]
                        health_blog = {
                            'title': article['title'],
                            'description': article['description'],
                            'url': article['url'],
                            'author': article['author'],
                            'image': article['urlToImage'],
                            'publishedAt': published_at
                        }
                        health_blogs.append(health_blog)
                    
                        # Extract tags from title and description, excluding stopwords
                        title_words = [word.lower() for word in article['title'].split() if word.lower() not in stop_words]
                        description_words = [word.lower() for word in article['description'].split() if word.lower() not in stop_words]
                        tags.extend(title_words)
                        tags.extend(description_words)

        # Count occurrences of each tag
        tag_counts = Counter(tags)
            
        # Select top 10 most common tags
        top_tags = tag_counts.most_common(10)

        # Pagination logic
        page = request.args.get('page', 1, type=int)
        per_page = 5  # Number of blog posts per page
        total_pages = ceil(len(health_blogs) / per_page)
        start_index = (page - 1) * per_page
        end_index = min(start_index + per_page, len(health_blogs))
        paginated_blogs = health_blogs[start_index:end_index]

        # Fetch Instagram feed using images from health blogs
        blog_posts, _, _ = get_health_blogs()
        recent_posts = blog_posts[:5]  # Displaying 5 recent posts
        instagram_feed = []
        for post in blog_posts[:6]:  # Displaying 6 images
            instagram_post = {
                'image': post['image'],
                'url': post['url']
            }
            instagram_feed.append(instagram_post)

        return render_template('blog.html', blog_posts=paginated_blogs, tags=top_tags, recent_posts=recent_posts, instagram_feed=instagram_feed,  pagination={'total_pages': total_pages, 'current_page': page})
    else:
        print(f"API request failed with status code: {response.status_code}")
        return render_template('blog.html', error_message="Failed to fetch search results. Please try again later.")

# Search route
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if query:
        url = f'https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}'
        response = requests.get(url)
        data = response.json()
        # Filter health-related articles
        health_blogs = []
        tags = []  # List to store tags
        for article in data.get('articles', [])[:10]:  # Limit to 10 articles
            if all(field in article for field in ['title', 'description', 'url', 'author', 'publishedAt', 'urlToImage']):
                if article['title'] and article['description'] and article['url'] and article['author'] and article['publishedAt'] and article['urlToImage']:
                        # Remove time part from publishedAt
                        published_at = article['publishedAt'].split('T')[0]
                        health_blog = {
                            'title': article['title'],
                            'description': article['description'],
                            'url': article['url'],
                            'author': article['author'],
                            'image': article['urlToImage'],
                            'publishedAt': published_at
                        }
                        health_blogs.append(health_blog)
                    
                        # Extract tags from title and description, excluding stopwords
                        title_words = [word.lower() for word in article['title'].split() if word.lower() not in stop_words]
                        description_words = [word.lower() for word in article['description'].split() if word.lower() not in stop_words]
                        tags.extend(title_words)
                        tags.extend(description_words)


        # Count occurrences of each tag
        tag_counts = Counter(tags)
            
        # Select top 10 most common tags
        top_tags = tag_counts.most_common(10)

        # Pagination logic
        page = request.args.get('page', 1, type=int)
        per_page = 5  # Number of blog posts per page
        total_pages = ceil(len(health_blogs) / per_page)
        start_index = (page - 1) * per_page
        end_index = min(start_index + per_page, len(health_blogs))
        paginated_blogs = health_blogs[start_index:end_index]

        # recent posts
        recent_posts = health_blogs[:5]
        # Fetch Instagram feed using images from health blogs
        instagram_feed = []
        for post in health_blogs[:6]:  # Displaying 6 images
            instagram_post = {
                'image': post['image'],
                'url': post['url']
            }
            instagram_feed.append(instagram_post)

        return render_template('blog.html', blog_posts=paginated_blogs, tags=top_tags, recent_posts=recent_posts, instagram_feed=instagram_feed,  pagination={'total_pages': total_pages, 'current_page': page})
    else:
        return redirect('/blog')

#  <---------------------------------------- blog.html page function (ends here) ----------------------------------------->
