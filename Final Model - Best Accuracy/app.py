from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

model_path = 'C:/Users/hp/Desktop/Project/PCOS Prediction/Model/model.pkl'
model = pickle.load(open(model_path, 'rb'))

X_columns = ['Period Length', 'Cycle Length', 'Age', 'Overweight', 
             'loss weight gain / weight loss', 'irregular or missed periods', 
             'Difficulty in conceiving', 'Hair growth on Chin', 'Hair growth  on Cheeks', 
             'Hair growth Between breasts', 'Hair growth  on Upper lips', 'Hair growth in Arms', 
             'Hair growth on Inner thighs', 'Acne or skin tags', 'Hair thinning or hair loss', 
             'Dark patches', 'always tired', 'more Mood Swings', 'exercise per week', 
             'eat outside per week', 'canned food often']

PCOS_check = {0: 'No', 1: 'Yes'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = {
            'Period Length': float(request.form['periodLength']),
            'Cycle Length': float(request.form['cycleLength']),
            'Age': float(request.form['age']),
            'Overweight': float(request.form['overweight']),
            'loss weight gain / weight loss': float(request.form['weightChange']),
            'irregular or missed periods': float(request.form['irregularPeriods']),
            'Difficulty in conceiving': float(request.form['conceivingDifficulty']),
            'Hair growth on Chin': float(request.form['chinHairGrowth']),
            'Hair growth  on Cheeks': float(request.form['cheekHairGrowth']),
            'Hair growth Between breasts': float(request.form['breastHairGrowth']),
            'Hair growth  on Upper lips': float(request.form['upperLipHairGrowth']),
            'Hair growth in Arms': float(request.form['armsHairGrowth']),
            'Hair growth on Inner thighs': float(request.form['innerThighsHairGrowth']),
            'Acne or skin tags': float(request.form['acneOrSkinTags']),
            'Hair thinning or hair loss': float(request.form['hairLoss']),
            'Dark patches': float(request.form['darkPatches']),
            'always tired': float(request.form['alwaysTired']),
            'more Mood Swings': float(request.form['moodSwings']),
            'exercise per week': float(request.form['exercisePerWeek']),
            'eat outside per week': float(request.form['eatOutsidePerWeek']),
            'canned food often': float(request.form['cannedFoodOften'])
        }

        input_data = pd.DataFrame([data])
        
        prediction = model.predict(input_data)

        result = PCOS_check[prediction[0]]
        return jsonify({"prediction_result": result})

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)




