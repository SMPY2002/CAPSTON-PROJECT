# MediScan360 ( Capston Project - 3rd Year )
MediScan360 Project is a comprehensive Flask-based web application designed for advanced healthcare diagnostics using machine learning models. It offers various health checkup tests based on uploaded data, providing accurate and personalized health insights.
## Features
### 1. Heart Attack Risk Assessment
Upload ECG data to assess the risk of a heart attack using Google's Gemini API for accurate analysis. This feature helps in early detection and prevention of cardiovascular diseases.
### 2. Thyroid Disease Identification
Predicts the presence and type of thyroid disease using a custom-built machine learning model trained on Kaggle's Thyroid Disease dataset. This model ensures reliable diagnosis and treatment recommendations.
### 3. Skin Disease Detection
Upload images of affected skin areas to detect skin diseases, including certain types of skin cancer. The model is based on research published in IEEE papers, ensuring high accuracy and sensitivity.
### 4. General Body Health Checkup
Utilizes Google's Gemini pre-trained model to analyze overall body health parameters such as blood pressure, cholesterol levels, and more. It provides users with a comprehensive health report and personalized suggestions for maintaining good health.
### 5. Resource Section
Displays health-related news and articles sourced from NewsAPI, ensuring users are informed about current health trends and developments.
## Technologies Used  
* __Flask__: Lightweight and flexible web framework for Python.
* __Machine Learning Models__: Leveraged Google's Gemini API and custom models trained on Kaggle datasets and research papers.
* __NewsAPI__: Integration for fetching real-time health-related news.
* __Development Environment__: Currently hosted on a development server.

## Installation and Usage
* Clone the repository:
  ```
  git clone https://github.com/your-username/MediScan360.git
  cd MediScan360
  ```
* Install dependencies:
  ```
  pip install -r requirements.txt
  ```
* Run the application:
  ```
  python app.py
  ```
* Accessing the Application:
   Open your browser and navigate to http://localhost:5000 to start using MediScan360.
## Dataset and Research References
* Thyroid Detection Dataset  --> ( https://www.kaggle.com/datasets/emmanuelfwerr/thyroid-disease-data )
* Skin Disease/Cancer Dataset --> ( https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?select=hmnist_28_28_RGB.csv )
* Research Paper on Skin Cancer --> ( https://ieeexplore.ieee.org/document/9397987 ) [ **Detection and Classification of Skin Cancer by Using a Parallel CNN Model** ]
## Screenshots 
Here are some pics of the website interface
![Screenshot 2024-07-18 212606](https://github.com/user-attachments/assets/a679b67d-faaf-419c-aacf-0ea456ebf29c)
  **Advance Test**
![Screenshot 2024-04-24 215502](https://github.com/user-attachments/assets/2215bc17-ff4d-49e7-abd0-30236210fee8)
  **Herat Attack Detection**
![Screenshot 2024-04-24 215106](https://github.com/user-attachments/assets/fea93d80-5a08-464b-96e0-16906d242b02)
  **Skin Disease Detection**
![Screenshot 2024-07-18 212459](https://github.com/user-attachments/assets/da081df5-dde0-4ac6-82e0-ef5b60c37d75)
  **Thyroid Detection**
![Screenshot 2024-04-24 215210](https://github.com/user-attachments/assets/ab06ab59-e40e-4a15-8a71-d68dce71f84c)
  **Health Briefs & Articles**
![Screenshot 2024-07-18 212352](https://github.com/user-attachments/assets/736de4fe-b976-463d-b900-c786071e5385)

![Screenshot 2024-07-18 212418](https://github.com/user-attachments/assets/96e34093-79b7-4c61-8215-4f49c311c9c2)

## Other Contributors
  * __Shivam Mishra__  [Github] --> { Shivam0000718 }
  * __Shubham Mishra__ [Github] --> { sbpy100 }

## Future Enhancements
* Incorporate more advanced machine learning models for enhanced diagnostics.
* Improve user interface and experience with modern design principles.
* Scale the application to a production environment for broader accessibility and usage.
