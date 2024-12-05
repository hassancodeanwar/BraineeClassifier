import React from 'react';
import { Brain, Shield, Clock, Activity } from 'lucide-react';

const features = [
  {
    icon: Brain,
    title: 'Advanced AI Analysis',
    description: 'State-of-the-art deep learning models for accurate MRI classification'
  },
  {
    icon: Shield,
    title: 'Secure Processing',
    description: 'Your medical data is processed with the highest security standards'
  },
  {
    icon: Clock,
    title: 'Rapid Results',
    description: 'Get analysis results within seconds'
  },
  {
    icon: Activity,
    title: 'High Accuracy',
    description: 'Validated against extensive medical datasets'
  }
];

export function Features() {
  return (
    <div className="bg-gray-50 py-12 sm:py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center space-y-2 sm:space-y-4">
          <h2 className="text-2xl sm:text-3xl font-bold text-gray-900">Why Choose Brainee?</h2>
          <p className="text-base sm:text-lg text-gray-600">
            Advanced technology meets medical precision
          </p>
        </div>

        <div className="mt-8 sm:mt-12 grid grid-cols-1 gap-6 sm:gap-8 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((feature, index) => (
            <div
              key={index}
              className="bg-white rounded-lg p-4 sm:p-6 shadow-sm hover:shadow-md transition-shadow"
            >
              <div className="flex flex-col items-center text-center space-y-3">
                <feature.icon className="h-6 w-6 sm:h-8 sm:w-8 text-blue-600" />
                <h3 className="text-base sm:text-lg font-semibold text-gray-900">
                  {feature.title}
                </h3>
                <p className="text-sm sm:text-base text-gray-600">{feature.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
} 