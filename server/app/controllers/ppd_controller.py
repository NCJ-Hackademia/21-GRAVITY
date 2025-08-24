from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.models.ppd_assessment import PPDAssessment
import pickle

# Load your trained PPD model
import os
ppd_model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'trained_ppd_model.pkl')
with open(ppd_model_path, 'rb') as f:
    ppd_model = pickle.load(f)

@jwt_required()
def assess_ppd():
    user_id = get_jwt_identity()
    data = request.get_json()
    
    # Predict using the ML model
    risk_score = ppd_model.predict_proba([data['responses']])[0][1]
    
    # Save assessment
    assessment = PPDAssessment.save_assessment(
        user_id, 
        data['responses'], 
        float(risk_score)
    )
    
    return jsonify({
        'risk_score': risk_score,
        'assessment_id': str(assessment.inserted_id)
    })

def get_latest_ppd_score(user_id):
    """Get the latest PPD assessment score for a user"""
    try:
        assessment = PPDAssessment.get_latest_assessment(user_id)
        if assessment:
            # Convert risk percentage to EPDS-like score
            epds_score = PPDAssessment.convert_risk_to_epds_score(assessment['risk_score'])
            return jsonify({
                'success': True,
                'score': epds_score,
                'risk_percentage': assessment['risk_score'],
                'date': assessment['date']
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No PPD assessment found'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
