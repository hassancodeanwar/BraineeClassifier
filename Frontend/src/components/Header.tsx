import React from 'react';
import { Brain } from 'lucide-react';

export function Header() {
  return (
    <header className="bg-white/80 backdrop-blur-md shadow-sm fixed w-full top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 sm:py-4">
        <div className="flex items-center justify-center md:justify-between">
          <div className="flex items-center space-x-2">
            <Brain className="h-6 w-6 sm:h-8 sm:w-8 text-blue-600" />
            <h1 className="text-xl sm:text-2xl font-bold text-gray-900 whitespace-nowrap">
              Brainee Classifier
            </h1>
          </div>
        </div>
      </div>
    </header>
  );
}
