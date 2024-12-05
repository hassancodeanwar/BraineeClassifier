# Brainee Classifier  

**Brainee Classifier** is a full-stack AI-powered system for brain tumor classification. It leverages deep learning with **VGG19 transfer learning** to classify medical images accurately and efficiently. The project includes a backend API built with **Flask** and a modern **React-based web application**, creating an interactive interface for users.  

---

## Features  
- **AI-Powered Classification**: Uses VGG19 transfer learning for high accuracy.  
- **Full-Stack Integration**: Flask backend connected to a React frontend for seamless user experience.  
- **Custom Data Pipeline**: Handles image preprocessing, augmentation, and loading.  
- **Advanced Training Optimization**: Incorporates callbacks like early stopping and learning rate adjustment.  
- **Interactive Visualizations**: Displays predictions, confusion matrices, and performance metrics.  
- **Deployment-Ready**: The model is saved as a reusable `.h5` file for real-world applications.  

---

## System Architecture  
1. **Data Processing**: Preprocessing, augmentation, and normalization of medical images.  
2. **Model Training**: Transfer learning with VGG19 and fine-tuning for enhanced performance.  
3. **Backend (Flask)**: Provides API endpoints for model inference and prediction.  
4. **Frontend (React)**: Interactive UI for uploading images and visualizing results.  

---

## Tech Stack  
### Backend:  
- Flask  
- TensorFlow/Keras  

### Frontend:  
- React.js  
- Axios  

### Machine Learning:  
- VGG19 Transfer Learning  
- Python Libraries: NumPy, Pandas, Matplotlib, Seaborn, Plotly, Scikit-learn  

---

## Installation  

### Prerequisites  
- Python 3.8+  
- Node.js and npm  

### Steps  
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/hassancodeanwar/braineeclassifier.git  
   cd brainee-classifier  
   ```  

2. **Backend Setup**  
   - Navigate to the `backend` folder:  
     ```bash
     cd backend  
     ```  
   - Install Python dependencies:  
     ```bash
     pip install -r requirements.txt  
     ```  
   - Run the Flask server:  
     ```bash
     python app.py  
     ```  

3. **Frontend Setup**  
   - Navigate to the `frontend` folder:  
     ```bash
     cd frontend  
     ```  
   - Install React dependencies:  
     ```bash
     npm install  
     ```  
   - Start the React app:  
     ```bash
     npm start  
     ```  

---

## Usage  
1. Upload a brain scan image through the React web app.  
2. The image is sent to the Flask backend for preprocessing and classification.  
3. The model predicts the tumor type and displays the result on the frontend.  

---

## Visualizations  
- **Model Performance**: Training/validation accuracy and loss curves.  
- **Confusion Matrix**: Displays prediction metrics for each class.  
- **Prediction Analysis**: Visual comparison of true vs. predicted labels.  

---

## Ethical Considerations  
- This tool is designed for research and educational purposes only.  
- It is not a substitute for professional medical advice or diagnosis.  
- Clinical validation is required before deployment in healthcare settings.  

---

## Future Enhancements  
- Support for additional tumor types and datasets.  
- Integration of more advanced AI models.  
- Improved UI/UX for user interaction.  

---

## License  
This project is licensed under the MIT License.  

---

## Acknowledgments  
- Pre-trained VGG19 model from ImageNet.  
- Inspiration from medical image classification research.  

For any questions or collaboration, feel free to reach out! 😊  
