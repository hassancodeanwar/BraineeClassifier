import React from 'react';
import { Brain, Github, Linkedin, Twitter } from 'lucide-react';

export function Footer() {
  return (
    <footer className="bg-white border-t">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        <div className="flex flex-col items-center space-y-6 sm:space-y-8">
          <div className="flex items-center space-x-2">
            <Brain className="h-6 w-6 sm:h-8 sm:w-8 text-blue-600" />
            <span className="text-xl sm:text-2xl font-bold text-gray-900">Brainee Classifier</span>
          </div>
          
          <div className="max-w-2xl text-center">
            <p className="text-sm sm:text-base text-gray-600">
              Brainee Classifier is a cutting-edge platform that leverages artificial intelligence 
              to assist medical professionals in analyzing brain MRI scans. Our mission is to make 
              advanced medical imaging analysis accessible and accurate.
            </p>
          </div>

          <div className="flex space-x-6">
            <a href="https://github.com/hassancodeanwar" target="_blank" rel="noopener noreferrer" 
               className="text-gray-400 hover:text-gray-500">
              <Github className="h-5 w-5 sm:h-6 sm:w-6" />
            </a>
            <a href="https://linkedin.com/hassancodeanwar" target="_blank" rel="noopener noreferrer"
               className="text-gray-400 hover:text-gray-500">
              <Linkedin className="h-5 w-5 sm:h-6 sm:w-6" />
            </a>
            {/* <a href="https://x.com/hassancodeanwar" target="_blank" rel="noopener noreferrer"
               className="text-gray-400 hover:text-gray-500">
              <Twitter className="h-5 w-5 sm:h-6 sm:w-6" />
            </a> */}
          </div>

          <div className="text-center text-xs sm:text-sm text-gray-500">
            <p>© {new Date().getFullYear()} Brainee Classifier. All rights reserved.</p>
            <p>For research and educational purposes only.</p>
          </div>
        </div>
      </div>
    </footer>
  );
}