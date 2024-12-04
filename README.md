# Brainee Classifier Web APP
---
## Project Aerchitechture
```bash
BraineeClassifier/
│
├── Backend/
│   ├── app.py (Size: 3,225 bytes)
│   ├── Brian_Tumor_Classifier.h5 (Size: 137 bytes)
│   ├── model.py (Size: 13,859 bytes)
│   ├── project_structure.txt (Size: 1,491 bytes)
│   ├── README.md (Size: 7,539 bytes)
│   ├── requirements.txt (Size: 91 bytes)
│   └── uploads/
│
├── WebApp/                      # Frontend application
│   ├── tailwind.config.js       # Tailwind CSS configuration
│   ├── tsconfig.app.json        # TypeScript configuration for the app
│   ├── tsconfig.node.json       # TypeScript configuration for Node.js
│   ├── tsconfig.json            # General TypeScript configuration
│   ├── eslint.config.js         # ESLint configuration
│   ├── postcss.config.js        # PostCSS configuration
│   ├── vite.config.ts           # Vite configuration
│   ├── package.json             # NPM dependencies
│   ├── package-lock.json        # Lock file for package versions
│   └── index.html               # HTML entry point
│
├── src/                         # Source files
│   ├── main.tsx                 # Entry point
│   ├── App.tsx                  # Main application component
│   ├── index.css                # Global styles
│   │
│   ├── types/                   # TypeScript type definitions
│   │   ├── file.ts
│   │   └── analysis.ts
│   │
│   ├── utils/                   # Utility functions
│   │   ├── reportGenerator.ts
│   │   ├── analysisResults.ts
│   │   └── fileHandling.ts
│   │
│   └── components/              # React components
│       ├── layout/              # Page layout components
│       │   ├── Header.tsx
│       │   ├── Footer.tsx
│       │   └── Link.tsx
│       │
│       ├── upload/              # File upload and analysis components
│       │   ├── UploadSection.tsx
│       │   ├── ResultCard.tsx
│       │   ├── RegionCard.tsx
│       │   └── AnalysisResults.tsx
│       │
│       └── home/                # Home page components
│           └── Hero.tsx
```


