from flask import Flask, render_template, request
from main import load_data, preprocess_data, train_model
import numpy as np
import os

app = Flask(__name__)

# Load and train the model at startup
data = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
model = train_model(X_train, X_test, y_train, y_test)

# Dictionary mapping crops to their descriptions
# Add this crop information dictionary
crop_info = {
    'rice': {
        'description': 'Rice is a staple food crop that thrives in warm, humid conditions. It requires significant water and proper soil nutrients. Ideal conditions include temperatures between 20-30°C and high humidity.',
        'care': 'Requires flooded fields, regular fertilization, and careful water management.'
    },
    'maize': {
        'description': 'Maize (corn) is a versatile grain crop that grows well in warm conditions. It needs full sunlight and moderate temperatures between 20-30°C.',
        'care': 'Needs well-drained soil, regular watering, and adequate spacing between plants.'
    },
    'chickpea': {
        'description': 'Chickpea is a protein-rich legume that adapts well to semi-arid conditions. It prefers temperatures between 15-25°C and moderate rainfall.',
        'care': 'Requires well-drained soil and moderate irrigation. Resistant to drought.'
    },
    'kidneybeans': {
        'description': 'Kidney beans thrive in warm, humid conditions. They need temperatures between 20-30°C and consistent moisture.',
        'care': 'Needs support for climbing, regular watering, and well-drained soil.'
    },
    'pigeonpeas': {
        'description': 'Pigeon peas are drought-resistant legumes that grow well in tropical conditions. They can tolerate high temperatures and various soil types.',
        'care': 'Minimal maintenance required, drought tolerant, and improves soil fertility.'
    },
    'mothbeans': {
        'description': 'Moth beans are heat and drought-tolerant legumes. They thrive in hot, dry conditions and can grow in poor soil.',
        'care': 'Very hardy, requires minimal irrigation, and suited for arid regions.'
    },
    'mungbean': {
        'description': 'Mung beans are fast-growing legumes that prefer warm temperatures. They are well-suited to tropical and subtropical regions.',
        'care': 'Needs good drainage, moderate water, and warm temperatures.'
    },
    'blackgram': {
        'description': 'Black gram is a drought-resistant pulse crop. It grows well in both tropical and subtropical conditions.',
        'care': 'Tolerates various soil types, needs moderate water, and minimal maintenance.'
    },
    'lentil': {
        'description': 'Lentils are cool-season legumes that prefer moderate temperatures. They are drought-tolerant and can grow in various soil types.',
        'care': 'Requires well-drained soil, moderate rainfall, and cool growing conditions.'
    },
    'pomegranate': {
        'description': 'Pomegranate is a fruit-bearing shrub that thrives in semi-arid conditions. It can tolerate both hot summers and mild winters.',
        'care': 'Needs good drainage, regular pruning, and protection from extreme cold.'
    },
    'banana': {
        'description': 'Bananas require tropical conditions with high humidity and temperatures. They need consistent moisture and rich soil.',
        'care': 'Requires regular watering, rich organic soil, and protection from strong winds.'
    },
    'mango': {
        'description': 'Mango trees thrive in tropical climates with distinct wet and dry seasons. They need high temperatures and good drainage.',
        'care': 'Needs protection from frost, good irrigation during flowering, and well-drained soil.'
    },
    'grapes': {
        'description': 'Grapes grow best in areas with long, warm summers and cool winters. They need full sun and good air circulation.',
        'care': 'Requires regular pruning, trellising, and protection from diseases.'
    },
    'watermelon': {
        'description': 'Watermelons need hot temperatures and long growing seasons. They require full sun and plenty of space to grow.',
        'care': 'Needs consistent moisture, rich soil, and good pollination for fruit development.'
    },
    'muskmelon': {
        'description': 'Muskmelons prefer hot, sunny conditions and well-drained soil. They need a long, warm growing season.',
        'care': 'Requires regular watering, good drainage, and protection from pests.'
    },
    'apple': {
        'description': 'Apple trees need a period of cold dormancy and moderate summers. They require well-drained soil and full sun.',
        'care': 'Needs regular pruning, pest management, and proper spacing between trees.'
    },
    'orange': {
        'description': 'Oranges thrive in subtropical climates with moderate temperatures. They need protection from freezing temperatures.',
        'care': 'Requires good drainage, regular fertilization, and protection from cold.'
    },
    'papaya': {
        'description': 'Papaya trees need tropical conditions with no frost. They grow quickly and produce fruit year-round in ideal conditions.',
        'care': 'Needs well-drained soil, regular watering, and protection from strong winds.'
    },
    'coconut': {
        'description': 'Coconut palms require tropical conditions with high humidity and temperatures. They need sandy soil and plenty of sunlight.',
        'care': 'Requires good drainage, regular watering, and protection from cold.'
    },
    'cotton': {
        'description': 'Cotton needs a long, hot growing season and plenty of sunshine. It requires well-drained soil and moderate rainfall.',
        'care': 'Needs regular pest monitoring, adequate moisture, and proper harvest timing.'
    },
    'jute': {
        'description': 'Jute grows best in humid, tropical conditions. It needs high rainfall and temperatures during the growing season.',
        'care': 'Requires fertile soil, good drainage, and consistent moisture.'
    },
    'coffee': {
        'description': 'Coffee plants thrive in tropical highlands with moderate temperatures. They need shade and well-drained soil.',
        'care': 'Requires shade management, regular pruning, and protection from frost.'
    },
    # Add this to your crop_info dictionary
    'onion': {
        'description': 'Onions are versatile root vegetables that grow well in moderate climates. They prefer temperatures between 13-25°C and require well-draining soil. Onions need full sun exposure and moderate moisture levels for optimal bulb development.',
        'care': 'Plant in rich, well-drained soil, maintain consistent moisture without waterlogging, and space plants 4-5 inches apart. Regular weeding is essential, and reduce watering when tops begin to fall over near harvest time.'
    },
}

# Update your route to include the crop information
@app.route('/', methods=['GET', 'POST'])
def home():
    predictions = None
    crop_details_list = None
    if request.method == 'POST':
        # Get values from the form
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorus'])
        K = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        # Make prediction
        user_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        user_input_scaled = scaler.transform(user_input)
        
        # Get probability scores for all crops
        probabilities = model.predict_proba(user_input_scaled)[0]
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        predictions = [(model.classes_[idx], probabilities[idx] * 100) for idx in top_3_indices]
        
        # Get crop details for top 3
        crop_details_list = [
            (crop, score, crop_info.get(crop.lower(), {}))
            for crop, score in predictions
        ]
    
    return render_template('index.html', predictions=crop_details_list)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
