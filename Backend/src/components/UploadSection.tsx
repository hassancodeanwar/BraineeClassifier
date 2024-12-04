import React, { useState } from 'react';
import { Upload, AlertCircle, Loader2 } from 'lucide-react';
// HI
type Classification = {
  prediction: string;
  confidence: number;
};

export function UploadSection() {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [classification, setClassification] = useState<Classification | null>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === 'dragenter' || e.type === 'dragover');
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const handleFile = (file: File) => {
    if (file.type.startsWith('image/')) {
      setSelectedFile(file);
      setClassification(null);
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result as string);
      reader.readAsDataURL(file);
    } else {
      alert('Please upload an image file');
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Unexpected error occurred');
      }

      const data = await response.json();
      setClassification({
        prediction: data.prediction,
        confidence: data.confidence,
      });
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to classify the image. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetUpload = () => {
    setSelectedFile(null);
    setPreview(null);
    setClassification(null);
  };

  return (
    <div id="upload-section" className="max-w-4xl mx-auto px-4 py-6 sm:py-8">
      <div
        className={`relative border-2 border-dashed rounded-lg p-4 sm:p-8 text-center ${
          dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          className="hidden"
          id="file-upload"
          accept="image/*"
          onChange={handleFileInput}
        />
        {!preview ? (
          <label
            htmlFor="file-upload"
            className="flex flex-col items-center cursor-pointer p-4 sm:p-6"
          >
            <Upload className="h-8 w-8 sm:h-12 sm:w-12 text-gray-400 mb-4" />
            <span className="text-sm sm:text-base text-gray-600">
              Drag and drop your MRI scan here, or{' '}
              <span className="text-blue-600">browse</span>
            </span>
            <span className="text-xs sm:text-sm text-gray-500 mt-2">
              Supported formats: JPEG, PNG
            </span>
          </label>
        ) : (
          <div className="space-y-4 sm:space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6 items-center">
              <div className="order-2 md:order-1">
                <img
                  src={preview}
                  alt="Preview"
                  className="max-h-48 sm:max-h-64 mx-auto rounded-lg"
                />
              </div>
              <div className="space-y-3 sm:space-y-4 order-1 md:order-2">
                {classification ? (
                  <div className="bg-green-50 p-3 sm:p-4 rounded-lg">
                    <h3 className="text-base sm:text-lg font-semibold text-green-800">
                      Analysis Results
                    </h3>
                    <p className="text-sm sm:text-base text-green-700 mt-2">
                      Classification: {classification.prediction}
                    </p>
                    <p className="text-sm sm:text-base text-green-700">
                      Confidence: {(classification.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                ) : (
                  <button
                    className={`w-full bg-blue-600 text-white px-4 sm:px-6 py-2 sm:py-3 rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center space-x-2 text-sm sm:text-base ${
                      isAnalyzing ? 'opacity-75 cursor-not-allowed' : ''
                    }`}
                    onClick={analyzeImage}
                    disabled={isAnalyzing}
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="h-4 w-4 sm:h-5 sm:w-5 animate-spin" />
                        <span>Analyzing...</span>
                      </>
                    ) : (
                      <span>Analyze Image</span>
                    )}
                  </button>
                )}
                <button
                  onClick={resetUpload}
                  className="w-full border border-gray-300 text-gray-600 px-4 sm:px-6 py-2 sm:py-3 rounded-lg hover:bg-gray-50 transition-colors text-sm sm:text-base"
                >
                  Upload Different Image
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
      <div className="mt-4 sm:mt-6 bg-blue-50 rounded-lg p-3 sm:p-4">
        <div className="flex items-start">
          <AlertCircle className="h-5 w-5 text-blue-600 mt-0.5 mr-2 flex-shrink-0" />
          <div className="text-xs sm:text-sm text-blue-800">
            <p className="mb-2">For accurate results, please ensure:</p>
            <ul className="list-disc ml-5">
              <li>The image is a clear brain MRI scan</li>
              <li>The scan is properly oriented</li>
              <li>The image is in high resolution</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
