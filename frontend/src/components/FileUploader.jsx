import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { Upload, File, X } from 'lucide-react';
import apiService from '../services/api';
import LoadingSpinner from './LoadingSpinner';

const FileUploader = ({ onSuccess, onError }) => {
    const [file, setFile] = useState(null);
    const [collectionName, setCollectionName] = useState('');
    const [loading, setLoading] = useState(false);
    const [dragActive, setDragActive] = useState(false);
    const fileInputRef = useRef(null);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileChange(e.dataTransfer.files[0]);
        }
    };

    const handleFileChange = (selectedFile) => {
        const allowedTypes = ['text/plain', 'application/pdf', 'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];

        if (selectedFile && allowedTypes.includes(selectedFile.type)) {
            setFile(selectedFile);
        } else {
            onError('Invalid file type. Please upload txt, pdf, doc, or docx files.');
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!file) {
            onError('Please select a file');
            return;
        }

        setLoading(true);

        const result = await apiService.generateFromFile(
            file,
            collectionName || 'test_ai_collection_v1'
        );

        setLoading(false);

        if (result.success) {
            onSuccess(`Question paper generated successfully! File: ${result.data.output_file}`);
            setFile(null);
            setCollectionName('');
        } else {
            onError(result.error);
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="generator-container"
        >
            <div className="card">
                <div className="card-header">
                    <Upload size={24} />
                    <h2>Generate from File</h2>
                </div>

                <form onSubmit={handleSubmit} className="form">
                    <div className="form-group">
                        <label>Upload Course Content File</label>
                        <div
                            className={`file-drop-zone ${dragActive ? 'active' : ''} ${file ? 'has-file' : ''}`}
                            onDragEnter={handleDrag}
                            onDragLeave={handleDrag}
                            onDragOver={handleDrag}
                            onDrop={handleDrop}
                            onClick={() => fileInputRef.current?.click()}
                        >
                            <input
                                ref={fileInputRef}
                                type="file"
                                onChange={(e) => handleFileChange(e.target.files[0])}
                                accept=".txt,.pdf,.doc,.docx"
                                style={{ display: 'none' }}
                                disabled={loading}
                            />

                            {file ? (
                                <div className="file-preview">
                                    <File size={48} className="file-icon" />
                                    <div className="file-info">
                                        <p className="file-name">{file.name}</p>
                                        <p className="file-size">{(file.size / 1024).toFixed(2)} KB</p>
                                    </div>
                                    <button
                                        type="button"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            setFile(null);
                                        }}
                                        className="file-remove"
                                        disabled={loading}
                                    >
                                        <X size={20} />
                                    </button>
                                </div>
                            ) : (
                                <div className="drop-zone-content">
                                    <Upload size={48} className="upload-icon" />
                                    <p className="drop-zone-text">
                                        Drag and drop your file here, or click to browse
                                    </p>
                                    <p className="drop-zone-hint">
                                        Supported formats: TXT, PDF, DOC, DOCX
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="form-group">
                        <label htmlFor="fileCollectionName">Collection Name (optional)</label>
                        <input
                            type="text"
                            id="fileCollectionName"
                            value={collectionName}
                            onChange={(e) => setCollectionName(e.target.value)}
                            placeholder="test_ai_collection_v1"
                            disabled={loading}
                        />
                    </div>

                    {loading ? (
                        <LoadingSpinner message="Uploading and generating..." />
                    ) : (
                        <motion.button
                            type="submit"
                            className="btn btn-primary"
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            disabled={!file}
                        >
                            <Upload size={20} />
                            Generate Question Paper
                        </motion.button>
                    )}
                </form>
            </div>
        </motion.div>
    );
};

export default FileUploader;
