import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FileText, Download, RefreshCw, Calendar, HardDrive } from 'lucide-react';
import apiService from '../services/api';
import LoadingSpinner from './LoadingSpinner';

const OutputsList = ({ onError }) => {
    const [outputs, setOutputs] = useState([]);
    const [loading, setLoading] = useState(false);

    const fetchOutputs = async () => {
        setLoading(true);
        const result = await apiService.getOutputs();
        setLoading(false);

        if (result.success) {
            setOutputs(result.data.files || []);
        } else {
            onError(result.error);
        }
    };

    useEffect(() => {
        fetchOutputs();
    }, []);

    const formatBytes = (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    };

    const formatDate = (dateString) => {
        return new Date(dateString).toLocaleString();
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
                    <FileText size={24} />
                    <h2>Generated Question Papers</h2>
                    <motion.button
                        onClick={fetchOutputs}
                        className="btn btn-secondary"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        disabled={loading}
                    >
                        <RefreshCw size={18} className={loading ? 'spinning' : ''} />
                        Refresh
                    </motion.button>
                </div>

                {loading ? (
                    <LoadingSpinner message="Loading outputs..." />
                ) : outputs.length === 0 ? (
                    <div className="empty-state">
                        <FileText size={64} className="empty-icon" />
                        <p className="empty-text">No question papers generated yet</p>
                        <p className="empty-hint">Generate your first question paper to see it here</p>
                    </div>
                ) : (
                    <div className="outputs-list">
                        {outputs.map((output, index) => (
                            <motion.div
                                key={output.filename}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: index * 0.1 }}
                                className="output-item"
                            >
                                <div className="output-icon">
                                    <FileText size={32} />
                                </div>
                                <div className="output-info">
                                    <h3 className="output-filename">{output.filename}</h3>
                                    <div className="output-meta">
                                        <span className="meta-item">
                                            <HardDrive size={14} />
                                            {formatBytes(output.size)}
                                        </span>
                                        <span className="meta-item">
                                            <Calendar size={14} />
                                            {formatDate(output.created)}
                                        </span>
                                    </div>
                                </div>
                                <motion.a
                                    href={apiService.downloadOutput(output.filename)}
                                    download
                                    className="btn btn-download"
                                    whileHover={{ scale: 1.05 }}
                                    whileTap={{ scale: 0.95 }}
                                >
                                    <Download size={18} />
                                    Download
                                </motion.a>
                            </motion.div>
                        ))}
                    </div>
                )}
            </div>
        </motion.div>
    );
};

export default OutputsList;
