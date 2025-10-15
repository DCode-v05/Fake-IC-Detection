import React from 'react';
import '../styles/Header.css';

/**
 * Header Component
 * Displays the application title and navigation
 */
function Header({ activeTab, onTabChange }) {
  return (
    <header className="header">
      <div className="header-container">
        <div className="logo-section">
          <div className="logo-icon">
            <img src="/download.png" alt="Company Logo" className="logo-image" />
          </div>
          <div className="title-section">
            <h1 className="title">IC Marking Detection</h1>
            <p className="subtitle">Automated Optical Inspection System</p>
          </div>
        </div>
        {/* Center brand text */}
        <div className="brand-center" aria-hidden>
          PixelAI
        </div>
        
        <nav className="nav">
          <button 
            className={`nav-button ${activeTab === 'detection' ? 'active' : ''}`}
            onClick={() => onTabChange('detection')}
          >
            <span className="nav-icon">🔍</span>
            Detect Fake IC
          </button>
          <button 
            className={`nav-button ${activeTab === 'validation' ? 'active' : ''}`}
            onClick={() => onTabChange('validation')}
          >
            <span className="nav-icon">✓</span>
            IC Validation
          </button>
        </nav>
      </div>
    </header>
  );
}

export default Header;
