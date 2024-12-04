# Brain Tumor Classifier API Documentation

## Overview
This Flask-based web application provides a machine learning service for brain tumor classification using a pre-trained deep learning model. The API allows users to upload medical brain scan images and receive predictions about the presence or absence of a tumor.

## System Requirements
- Python 3.10.14
- Flask
- TensorFlow
- NumPy
- Werkzeug
- Flask-CORS

## Deployment Configuration

### Environment Setup
1. Create a virtual environment
2. Install required dependencies:
```bash
pip install flask flask-cors tensorflow numpy werkzeug
```

### File Structure
```
Backend/
│
├── Brain_Tumor_Classifier_Enhanced.h5  # Pre-trained model
├── app.py                              # Main application file
└── uploads/                            # Temporary image upload directory
```

## API Endpoints

### 1. Image Prediction Endpoint
- **Route**: `/predict`
- **Method**: POST
- **Description**: Uploads and classifies a brain scan image

#### Request Parameters
- `file`: Medical brain scan image (PNG, JPG, JPEG)
- **Max File Size**: 5 MB
- **Allowed Extensions**: .png, .jpg, .jpeg

#### Response
```json
{
    "prediction": "Tumor" | "No Tumor",
    "confidence": 0.85  // Prediction confidence (0-1)
}
```

#### Error Responses
- `400`: Invalid file upload
- `500`: Image processing error

### 2. Health Check Endpoint
- **Route**: `/health`
- **Method**: GET
- **Description**: Checks API service status

#### Response
```json
{
    "status": "healthy"
}
```

## Image Processing Workflow
1. File validation
2. Secure filename generation
3. Image preprocessing
   - Resize to 512x512 pixels
   - Normalize pixel values (0-1)
4. Model prediction
5. Confidence calculation
6. Temporary file cleanup

## Security Considerations
- File type validation
- File size limitation
- Secure filename handling
- Temporary file removal after processing

## Model Details
- **Model Type**: Deep Learning (Convolutional Neural Network)
- **Input Shape**: 512 x 512 x 3 (RGB image)
- **Output**: Binary classification (Tumor/No Tumor)
- **Preprocessing**: Pixel normalization

## Deployment Instructions
```bash
# Run the application
python app.py

# Default Access
# http://localhost:5000
```

## Logging and Error Handling
- Errors are printed to console
- Comprehensive error responses for client
- No sensitive information exposed in error messages

## Performance Notes
- Inference time depends on model complexity
- Recommended for low to moderate traffic
- Consider horizontal scaling for high-load scenarios

## Potential Improvements
- Add more detailed logging
- Implement rate limiting
- Add authentication
- Support batch predictions
- Enhance error handling

## Troubleshooting
- Ensure model file `Brain_Tumor_Classifier_Enhanced.h5` is present
- Check TensorFlow and Keras versions compatibility
- Verify image preprocessing matches training configuration

## Disclaimer
This is a medical assistance tool and should not replace professional medical diagnosis. Always consult healthcare professionals.