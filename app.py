
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pathlib import Path
import tempfile

# Import your existing fraud detection system
from complete_system import CompleteFraudSystem

# Initialize FastAPI
app = FastAPI(
    title="Aadhaar Fraud Detection API",
    description="AI-powered fraud detection with 2-layer analysis",
    version="1.0.0"
)

# Enable CORS (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize fraud detection system (load once at startup)
fraud_system = None

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global fraud_system
    print("🔄 Loading fraud detection system...")
    fraud_system = CompleteFraudSystem(
        cnn_model_path='models/fraud_cnn.h5',
        img_size=(128, 128)
    )
    print("✅ System ready!")


@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "message": "Aadhaar Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/analyze (POST)",
            "health": "/health (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "fraud_detector": "loaded" if fraud_system else "not loaded",
        "cnn_model": "loaded" if fraud_system and fraud_system.cnn_model else "not loaded"
    }


@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Upload Aadhaar image and get complete fraud analysis
    
    Args:
        file: Image file (JPG, PNG)
    
    Returns:
        Complete fraud analysis report with recommendation
    """
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPG, PNG)")
    
    # Validate file extension
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(400, f"Invalid file type. Allowed: {allowed_extensions}")
    
    # Create temporary file to save upload
    temp_file = None
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file = tmp.name
        
        # Run fraud detection
        report = fraud_system.analyze_complete(temp_file, verbose=False)
        
        # Clean up report for JSON serialization
        response = {
            "status": "success",
            "filename": file.filename,
            "analysis": {
                "layer1_algorithmic": {
                    "overall_risk": report['layer1_algorithmic']['overall_risk'],
                    "fraud_detected": report['layer1_algorithmic']['fraud_detected'],
                    "risk_counts": report['layer1_algorithmic']['risk_counts'],
                    "techniques": {
                        "ela": {
                            "risk": report['layer1_algorithmic']['techniques']['ela']['risk'],
                            "confidence": report['layer1_algorithmic']['techniques']['ela']['confidence']
                        },
                        "metadata": {
                            "risk": report['layer1_algorithmic']['techniques']['metadata']['risk'],
                            "confidence": report['layer1_algorithmic']['techniques']['metadata']['confidence']
                        },
                        "overlay": {
                            "risk": report['layer1_algorithmic']['techniques']['overlay']['risk'],
                            "confidence": report['layer1_algorithmic']['techniques']['overlay']['confidence']
                        },
                        "fonts": {
                            "risk": report['layer1_algorithmic']['techniques']['fonts']['risk'],
                            "confidence": report['layer1_algorithmic']['techniques']['fonts']['confidence']
                        }
                    }
                },
                "layer2_cnn": {
                    "available": report['layer2_cnn']['available'],
                    "prediction": report['layer2_cnn'].get('prediction', 'N/A'),
                    "fake_probability": float(report['layer2_cnn'].get('fake_probability', 0)),
                    "authentic_probability": float(report['layer2_cnn'].get('authentic_probability', 0)),
                    "confidence": float(report['layer2_cnn'].get('confidence', 0))
                },
                "final_decision": report['final_decision']
            }
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass


@app.post("/analyze/quick")
async def quick_analyze(file: UploadFile = File(...)):
    """
    Quick analysis - Returns only final decision
    Faster endpoint for basic checks
    """
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    temp_file = None
    
    try:
        # Save upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file = tmp.name
        
        # Analyze
        report = fraud_system.analyze_complete(temp_file, verbose=False)
        
        # Return simplified response
        return {
            "status": "success",
            "decision": report['final_decision']['decision'],
            "confidence": report['final_decision']['confidence'],
            "reasoning": report['final_decision']['reasoning']
        }
    
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass


# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
