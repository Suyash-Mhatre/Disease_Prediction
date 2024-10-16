from flask import Flask, render_template, request
import pickle
import numpy as np
from scipy import stats

# Load the pre-trained model
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))
model3 = pickle.load(open('model3.pkl', 'rb'))

app = Flask(__name__)

symptoms = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
    'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
    'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets',
    'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level',
    'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion',
    'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
    'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
    'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes',
    'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes',
    'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
    'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain',
    'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
    'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
    'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness',
    'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance',
    'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression',
    'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation',
    'dischromic_patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history',
    'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf',
    'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
    'yellow_crust_ooze'
]

@app.route('/')
def index():
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [int(request.form.get(symptom, 0)) for symptom in symptoms]
    
    if all(value == 0 for value in input_data):
        result = "Choose one option / You have no Disease"
    elif all(value == 1 for value in input_data):
            result = "You are Lucky to be alive"
    else:
        input_data = np.array(input_data).reshape(1, -1)
        
        prediction1 = model1.predict(input_data)
        prediction1 = int(prediction1[0])
        
        prediction2 = model2.predict(input_data)
        prediction2 = int(prediction2[0])
        
        prediction3 = model3.predict(input_data)
        prediction3 = int(prediction3[0])
        
        all_predictions = [prediction1, prediction2, prediction3]
        
        mode_result = stats.mode(all_predictions)
        mode_result = mode_result.mode
        
        result_mapping = {
            0: '(vertigo) Paroymsal Positional Vertigo',
            1: 'AIDS',
            2: 'Acne',
            3: 'Alcoholic hepatitis',
            4: 'Allergy',
            5: 'Arthritis',
            6: 'Bronchial Asthma',
            7: 'Cervical spondylosis',
            8: 'Chicken pox',
            9: 'Chronic cholestasis',
            10: 'Common Cold',
            11: 'Dengue',
            12: 'Diabetes',
            13: 'Dimorphic hemmorhoids(piles)',
            14: 'Drug Reaction',
            15: 'Fungal infection',
            16: 'GERD',
            17: 'Gastroenteritis',
            18: 'Heart attack',
            19: 'Hepatitis B',
            20: 'Hepatitis C',
            21: 'Hepatitis D',
            22: 'Hepatitis E',
            23: 'Hypertension',
            24: 'Hyperthyroidism',
            25: 'Hypoglycemia',
            26: 'Hypothyroidism',
            27: 'Impetigo',
            28: 'Jaundice',
            29: 'Malaria',
            30: 'Migraine',
            31: 'Osteoarthristis',
            32: 'Paralysis (brain hemorrhage)',
            33: 'Peptic ulcer disease',
            34: 'Pneumonia',
            35: 'Psoriasis',
            36: 'Tuberculosis',
            37: 'Typhoid',
            38: 'Urinary tract infection',
            39: 'Varicose veins',
            40: 'hepatitis A'
        }
        
        result = result_mapping.get(mode_result, 'Unknown')
    
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
