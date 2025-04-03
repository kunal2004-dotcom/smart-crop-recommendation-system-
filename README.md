# Smart Crop Recommendation System

An AI-powered web application that helps farmers make informed decisions about crop selection based on soil conditions and environmental parameters.

## Features

- Real-time crop recommendations based on soil and environmental data
- Interactive voice assistant for guidance
- Top 3 crop recommendations with confidence scores
- Detailed crop information and care instructions
- Visual representation of recommended crops
- Voice-enabled results reading

## Parameters Required

1. **Soil Parameters**
   - Nitrogen (N) - mg/kg
   - Phosphorus (P) - mg/kg
   - Potassium (K) - mg/kg
   - pH Level

2. **Environmental Parameters**
   - Temperature (°C)
   - Humidity (%)
   - Rainfall (mm)

## Supported Crops

The system can recommend various crops including:
- Apple
- Banana
- Grapes
- Mango
- Muskmelon
- Orange
- Pomegranate
- Watermelon

## Technologies Used

- Frontend: HTML, CSS, JavaScript
- Backend: Python (Flask)
- Machine Learning: Scikit-learn
- Speech Synthesis: Web Speech API

## Setup Instructions

1. Install Python requirements:
```bash
pip install -r requirements.txt
'''

2. Run the Flask application:
```bash
python app.py
 ```

3. Access the application at http://localhost:5000

## Usage
1. Enter the required soil and environmental parameters in the input form
2. Click "Get Recommendation" to receive AI-powered crop suggestions
3. View the top 3 recommended crops with their confidence scores
4. Click "Hear Recommendations" to listen to the results

## Project Structure
croprecom/
├── static/
│   ├── style.css
│   ├── assistant.png
│   ├── speaker.png
│   └── crops/
├── templates/
│   └── index.html
├── crop_recommendation.csv
├── app.py
└── README.md

## Data Source
The model is trained on agricultural data containing various crop parameters and their optimal growing conditions.

## Contributing
Feel free to submit issues and enhancement requests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
