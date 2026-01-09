import { useState } from 'react';
import { AnimatePresence } from 'framer-motion';
import Header from './components/Header';
import TextGenerator from './components/TextGenerator';
import FileUploader from './components/FileUploader';
import OutputsList from './components/OutputsList';
import Toast from './components/Toast';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('text');
  const [toast, setToast] = useState(null);

  const showToast = (message, type = 'success') => {
    setToast({ message, type });
  };

  const handleSuccess = (message) => {
    showToast(message, 'success');
  };

  const handleError = (message) => {
    showToast(message, 'error');
  };

  return (
    <div className="app">
      <Header />

      <main className="main-content">
        <div className="container">
          <div className="tabs">
            <button
              className={`tab ${activeTab === 'text' ? 'active' : ''}`}
              onClick={() => setActiveTab('text')}
            >
              Generate from Text
            </button>
            <button
              className={`tab ${activeTab === 'file' ? 'active' : ''}`}
              onClick={() => setActiveTab('file')}
            >
              Generate from File
            </button>
            <button
              className={`tab ${activeTab === 'outputs' ? 'active' : ''}`}
              onClick={() => setActiveTab('outputs')}
            >
              View Outputs
            </button>
          </div>

          <AnimatePresence mode="wait">
            {activeTab === 'text' && (
              <TextGenerator
                key="text"
                onSuccess={handleSuccess}
                onError={handleError}
              />
            )}
            {activeTab === 'file' && (
              <FileUploader
                key="file"
                onSuccess={handleSuccess}
                onError={handleError}
              />
            )}
            {activeTab === 'outputs' && (
              <OutputsList
                key="outputs"
                onError={handleError}
              />
            )}
          </AnimatePresence>
        </div>
      </main>

      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
    </div>
  );
}

export default App;
