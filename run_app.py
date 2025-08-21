"""
Quick launcher script for the Blood Cell AI application
"""

import os
import sys

def main():
    # Add current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import and run the main application
    try:
        from main import app
        print("🚀 Starting Blood Cell AI Classification System...")
        print("📱 Open your browser to: http://localhost:5000")
        print("🔧 Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Run the Flask application
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        print(f"❌ Error importing main application: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    main()
