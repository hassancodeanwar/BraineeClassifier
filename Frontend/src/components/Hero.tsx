import React from 'react';
import { ArrowDown } from 'lucide-react';

export function Hero() {
  const scrollToUpload = () => {
    const uploadSection = document.getElementById('upload-section');
    uploadSection?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="relative bg-gradient-to-b from-blue-50 to-white min-h-screen flex items-center justify-center">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full py-12 sm:py-16 lg:py-20">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-12 items-center">
          <div className="text-center md:text-left space-y-6 md:space-y-8">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl xl:text-7xl font-bold text-gray-900 leading-tight">
              Advanced Brain MRI Analysis Using AI
            </h1>
            <p className="text-lg sm:text-xl lg:text-2xl text-gray-600 max-w-2xl">
              Get instant, accurate classifications for brain MRI scans using our state-of-the-art AI technology.
            </p>
            <button
              onClick={scrollToUpload}
              className="bg-blue-600 text-white px-6 sm:px-8 py-3 sm:py-4 rounded-lg hover:bg-blue-700 transition-colors inline-flex items-center space-x-3 text-base sm:text-lg shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-200"
            >
              <span>Start Analysis</span>
              <ArrowDown className="h-5 w-5" />
            </button>
          </div>
          <div className="relative flex justify-center md:justify-end mt-8 md:mt-0">
            <img
              src="https://img.freepik.com/premium-photo/understanding-complexity-human-brain-anatomy-exploring-regions-functions_578399-6191.jpg"
              alt="Brain MRI Visualization"
              className="rounded-2xl shadow-2xl w-full max-w-lg object-cover transform hover:scale-105 transition-transform duration-300"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
