#!/usr/bin/env python3
"""
Sinhala LLM Web Interface
Flask web application for easy interaction with the Sinhala LLM model
"""

from flask import Flask, render_template, request, jsonify, stream_template
import threading
import time
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from use_model import SinhalaLLMInference

app = Flask(__name__)
app.secret_key = 'sinhala_llm_secret_key'

# Global model instance
model = None
model_loading = False
model_loaded = False
model_error = None

def load_model():
    """Load the model in a separate thread"""
    global model, model_loading, model_loaded, model_error
    
    model_loading = True
    model_loaded = False
    model_error = None
    
    try:
        print("Loading Sinhala LLM model...")
        # Get the directory of this script file and build path to model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, '..', '..', 'models', 'sinhala_llm_best.pt')
        model = SinhalaLLMInference(model_path=model_path)
        model_loaded = True
        model_loading = False
        print("Model loaded successfully!")
    except Exception as e:
        model_error = str(e)
        model_loading = False
        model_loaded = False
        print(f"Error loading model: {e}")

# Start loading model when the app starts
threading.Thread(target=load_model, daemon=True).start()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/status')
def status():
    """Check model loading status"""
    global model_loading, model_loaded, model_error
    
    return jsonify({
        'loading': model_loading,
        'loaded': model_loaded,
        'error': model_error
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Generate text from prompt"""
    global model, model_loaded
    
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded yet. Please wait and try again.'
        }), 400
    
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        max_length = int(data.get('max_length', 100))
        temperature = float(data.get('temperature', 0.8))
        top_p = float(data.get('top_p', 0.9))
        
        if not prompt:
            return jsonify({'error': 'Please enter a prompt'}), 400
        
        # Validate parameters
        if max_length < 10 or max_length > 500:
            return jsonify({'error': 'Max length must be between 10 and 500'}), 400
        
        if temperature < 0.1 or temperature > 2.0:
            return jsonify({'error': 'Temperature must be between 0.1 and 2.0'}), 400
        
        if top_p < 0.1 or top_p > 1.0:
            return jsonify({'error': 'Top-p must be between 0.1 and 1.0'}), 400
        
        # Generate text
        generated_text = model.generate_text(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
        
        return jsonify({
            'generated_text': generated_text,
            'input_prompt': prompt
        })
        
    except Exception as e:
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Chat mode endpoint"""
    global model, model_loaded
    
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded yet. Please wait and try again.'
        }), 400
    
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Please enter a message'}), 400
        
        # Format as conversation
        formatted_prompt = f"<|user|>{message}<|assistant|>"
        
        # Generate response
        response = model.generate_text(
            prompt=formatted_prompt,
            max_length=200,
            temperature=0.8,
            top_p=0.9,
            do_sample=True
        )
        
        # Extract assistant response
        if "<|assistant|>" in response:
            assistant_response = response.split("<|assistant|>")[-1]
            if "<|end|>" in assistant_response:
                assistant_response = assistant_response.split("<|end|>")[0]
            assistant_response = assistant_response.strip()
        else:
            assistant_response = response
        
        return jsonify({
            'response': assistant_response,
            'user_message': message
        })
        
    except Exception as e:
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500

@app.route('/complete', methods=['POST'])
def complete():
    """Text completion endpoint"""
    global model, model_loaded
    
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded yet. Please wait and try again.'
        }), 400
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        max_length = int(data.get('max_length', 100))
        
        if not text:
            return jsonify({'error': 'Please enter text to complete'}), 400
        
        # Complete text
        completed_text = model.complete_text(text, max_length=max_length)
        
        return jsonify({
            'completed_text': completed_text,
            'original_text': text
        })
        
    except Exception as e:
        return jsonify({'error': f'Completion failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000) 