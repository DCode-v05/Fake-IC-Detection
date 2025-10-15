# IC Marking Detection - Frontend

Professional React.js frontend for the Automated Optical Inspection (AOI) based IC Marking Detection System.

## Features

- **Modern UI Design**: Professional, industry-appropriate interface with no emojis
- **Image Upload**: Drag-and-drop or click-to-upload functionality
- **Real-time Preview**: Instant preview of uploaded IC images
- **Inspection Dashboard**: Comprehensive controls and status monitoring
- **Modular Architecture**: Easy to debug and maintain
- **Responsive Design**: Works seamlessly across devices

## Project Structure

```
frontend/
├── public/
│   └── index.html              # HTML template
├── src/
│   ├── components/             # Modular React components
│   │   ├── Header.js           # Application header with navigation
│   │   ├── ImageUpload.js      # Image upload component with drag-and-drop
│   │   └── InspectionDashboard.js  # Inspection controls and results display
│   ├── styles/                 # Component-specific CSS files
│   │   ├── App.css             # Global styles and variables
│   │   ├── Header.css          # Header component styles
│   │   ├── ImageUpload.css     # Image upload component styles
│   │   └── InspectionDashboard.css  # Dashboard component styles
│   ├── App.js                  # Main application component
│   └── index.js                # Application entry point
├── package.json                # Dependencies and scripts
└── README.md                   # This file
```

## Installation

1. Navigate to the frontend directory:
```cmd
cd "d:\Deni\Mr. Tech\AI\Projects\Fake IC Marking Detection\frontend"
```

2. Install dependencies:
```cmd
npm install
```

## Running the Application

Start the development server:
```cmd
npm start
```

The application will open in your browser at `http://localhost:3000`

## Available Scripts

- `npm start` - Runs the app in development mode
- `npm build` - Builds the app for production
- `npm test` - Launches the test runner

## Component Overview

### Header Component
- Displays application title and branding
- Navigation menu for different sections
- Sticky header with modern gradient design

### ImageUpload Component
- Drag-and-drop file upload
- Click-to-browse functionality
- Image preview with file information
- File validation (type and size)
- Clear/reset functionality

### InspectionDashboard Component
- Image analysis information
- Inspection controls with status indicators
- Configurable inspection options
- Results display panel
- Progress indicators for processing

## Design Philosophy

- **Professional**: Clean, industrial-themed interface suitable for manufacturing/QA environments
- **User-Friendly**: Intuitive controls and clear visual feedback
- **Modular**: Each component is self-contained for easy maintenance and debugging
- **Responsive**: Adapts to different screen sizes and devices

## Color Scheme

The UI uses an industrial/technical color palette:
- Primary: Electric Blue (#00D4FF)
- Background: Dark Navy (#0F1419, #1A1F2E)
- Accent: Success Green (#00C853), Warning Orange (#FF9800), Error Red (#D32F2F)

## Future Integration

The frontend is designed to integrate with a Django backend API. The `InspectionDashboard` component includes placeholder API calls that will connect to the backend for:
- Image processing
- IC marking detection
- OEM database comparison
- Result generation

## Technology Stack

- React.js 18.2.0
- React DOM 18.2.0
- React Scripts 5.0.1
- CSS3 with CSS Variables

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)
