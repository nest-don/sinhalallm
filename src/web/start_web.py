#!/usr/bin/env python3
"""
Start Sinhala LLM Web Interface
Simple script to start the web interface with proper configuration
"""

import os
import sys
from web_interface import app

def main():
    """Main function to start the web interface"""
    print("🇱🇰 Starting Sinhala LLM Web Interface...")
    print("=" * 50)
    
    # Get the directory of this script file and build paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', '..', 'models', 'sinhala_llm_best.pt')
    tokenizer_path = os.path.join(script_dir, '..', '..', 'models', 'tokenizer', 'sinhala_sp.model')
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print("❌ Error: Model file 'sinhala_llm_best.pt' not found!")
        print("   Make sure the model file is in the models directory.")
        sys.exit(1)
    
    # Check if tokenizer exists
    if not os.path.exists(tokenizer_path):
        print("❌ Error: Tokenizer file 'sinhala_sp.model' not found!")
        print("   Make sure the tokenizer file is in the models/tokenizer directory.")
        sys.exit(1)
    
    print(f"✅ Model file found: {model_path}")
    print(f"✅ Tokenizer file found: {tokenizer_path}")
    print()
    
    # Configuration
    host = '0.0.0.0'  # Allow connections from any IP
    port = 3000  # Changed from 5000 to avoid macOS AirPlay conflict
    debug = False  # Set to True for development
    
    print(f"🌐 Starting web server on http://localhost:{port}")
    print(f"📱 Also accessible on your network at http://<your-ip>:{port}")
    print()
    print("Features available:")
    print("  📝 Text Generation - Generate creative Sinhala text")
    print("  💬 Chat Mode - Interactive conversation")
    print("  ✏️  Text Completion - Complete partial text")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")

if __name__ == "__main__":
    main() 