import { useState } from 'react';
import { motion } from 'framer-motion';
import { FileText, Send, Zap } from 'lucide-react';
import apiService from '../services/api';
import LoadingSpinner from './LoadingSpinner';
import StreamingDisplay, { useStreaming } from './StreamingDisplay';

const TextGenerator = ({ onSuccess, onError }) => {
    const [text, setText] = useState('');
    const [collectionName, setCollectionName] = useState('');
    const [loading, setLoading] = useState(false);
    const [enableStreaming, setEnableStreaming] = useState(true);

    const {
        events,
        isStreaming,
        isComplete,
        error: streamError,
        result: streamResult,
        startStreaming,
    } = useStreaming('http://localhost:5000/api/generate-stream', {
        onComplete: (data) => {
            onSuccess(`Question paper generated successfully! File: ${data.output_file}`);
            setText('');
            setCollectionName('');
        },
        onError: (error) => {
            onError(error);
        },
    });

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!text.trim()) {
            onError('Please enter course content');
            return;
        }

        if (enableStreaming) {
            // Use streaming mode
            startStreaming({
                text: text,
                collection_name: collectionName || 'test_ai_collection_v1'
            });
        } else {
            // Use non-streaming mode
            setLoading(true);

            const result = await apiService.generateFromText(
                text,
                collectionName || 'test_ai_collection_v1'
            );

            setLoading(false);

            if (result.success) {
                onSuccess(`Question paper generated successfully! File: ${result.data.output_file}`);
                setText('');
                setCollectionName('');
            } else {
                onError(result.error);
            }
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
                    <FileText size={24} />
                    <h2>Generate from Text</h2>
                </div>

                <form onSubmit={handleSubmit} className="form">
                    <div className="form-group">
                        <label htmlFor="courseText">Course Content</label>
                        <textarea
                            id="courseText"
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            placeholder="Paste your course content here...

Example:
Course: Programming Fundamentals

Course Outcomes:
CO1: Understand basic programming concepts
CO2: Apply programming constructs to solve problems

Syllabus Content:
Unit 1: Introduction to programming
Unit 2: Control structures
..."
                            rows={12}
                            disabled={loading || isStreaming}
                        />
                    </div>

                    <div className="form-group">
                        <label htmlFor="collectionName">Collection Name (optional)</label>
                        <input
                            type="text"
                            id="collectionName"
                            value={collectionName}
                            onChange={(e) => setCollectionName(e.target.value)}
                            placeholder="test_ai_collection_v1"
                            disabled={loading || isStreaming}
                        />
                    </div>

                    <div className="form-group">
                        <label className="streaming-toggle">
                            <input
                                type="checkbox"
                                checked={enableStreaming}
                                onChange={(e) => setEnableStreaming(e.target.checked)}
                                disabled={loading || isStreaming}
                            />
                            <span className="toggle-label">
                                <Zap size={16} />
                                Show AI Thinking Process (Streaming)
                            </span>
                        </label>
                    </div>

                    {isStreaming && (
                        <StreamingDisplay
                            taskId={collectionName || 'test_ai_collection_v1'}
                            events={events}
                            isComplete={isComplete}
                            hasError={!!streamError}
                        />
                    )}

                    {loading ? (
                        <LoadingSpinner message="Generating question paper..." />
                    ) : !isStreaming && (
                        <motion.button
                            type="submit"
                            className="btn btn-primary"
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                        >
                            <Send size={20} />
                            Generate Question Paper
                        </motion.button>
                    )}
                </form>
            </div>
        </motion.div>
    );
};

export default TextGenerator;
