import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2, CheckCircle, XCircle, Sparkles, Copy, ChevronDown, ChevronUp } from 'lucide-react';

const StreamingDisplay = ({ events = [], isComplete = false, hasError = false }) => {
    const [isExpanded, setIsExpanded] = useState(true);
    const eventsEndRef = useRef(null);

    const scrollToBottom = () => {
        eventsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [events]);

    const copyToClipboard = () => {
        const text = events.map(e => e.message).join('\n');
        navigator.clipboard.writeText(text);
    };

    const getEventIcon = (event) => {
        if (event.message && (event.message.includes('✅') || event.message.includes('complete'))) {
            return <CheckCircle size={16} className="event-icon success" />;
        } else if (event.message && (event.message.includes('❌') || event.message.includes('error'))) {
            return <XCircle size={16} className="event-icon error" />;
        } else {
            return <Sparkles size={16} className="event-icon thinking" />;
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="streaming-display"
        >
            <div className="streaming-header">
                <div className="streaming-title">
                    {!isComplete && !hasError && (
                        <Loader2 size={20} className="spinning" />
                    )}
                    {isComplete && <CheckCircle size={20} className="success-icon" />}
                    {hasError && <XCircle size={20} className="error-icon" />}
                    <span>
                        {isComplete ? 'Generation Complete!' : hasError ? 'Error Occurred' : 'Generating Question Paper...'}
                    </span>
                </div>
                <div className="streaming-actions">
                    <button onClick={copyToClipboard} className="icon-btn" title="Copy thinking log">
                        <Copy size={16} />
                    </button>
                    <button onClick={() => setIsExpanded(!isExpanded)} className="icon-btn">
                        {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </button>
                </div>
            </div>

            <AnimatePresence>
                {isExpanded && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="streaming-content"
                    >
                        <div className="events-list">
                            <AnimatePresence>
                                {events.map((event, index) => (
                                    <motion.div
                                        key={index}
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: index * 0.05 }}
                                        className="event-item"
                                    >
                                        {getEventIcon(event)}
                                        <span className="event-message">{event.message}</span>
                                    </motion.div>
                                ))}
                            </AnimatePresence>
                            <div ref={eventsEndRef} />
                        </div>

                        {!isComplete && !hasError && (
                            <div className="thinking-indicator">
                                <div className="dot"></div>
                                <div className="dot"></div>
                                <div className="dot"></div>
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
};

// Hook to handle EventSource streaming
export const useStreaming = (url, options = {}) => {
    const [events, setEvents] = useState([]);
    const [isStreaming, setIsStreaming] = useState(false);
    const [isComplete, setIsComplete] = useState(false);
    const [error, setError] = useState(null);
    const [result, setResult] = useState(null);
    const eventSourceRef = useRef(null);

    const startStreaming = (requestData) => {
        setEvents([]);
        setIsStreaming(true);
        setIsComplete(false);
        setError(null);
        setResult(null);

        // For POST requests with JSON, we need to use fetch with streaming
        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData),
        })
            .then(response => {
                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                const readStream = () => {
                    reader.read().then(({ done, value }) => {
                        if (done) {
                            setIsStreaming(false);
                            return;
                        }

                        const chunk = decoder.decode(value);
                        const lines = chunk.split('\n');

                        lines.forEach(line => {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.substring(6));

                                    if (data.event === 'thinking') {
                                        setEvents(prev => [...prev, { message: data.message, type: 'thinking' }]);
                                    } else if (data.event === 'complete') {
                                        setEvents(prev => [...prev, { message: data.message, type: 'success' }]);
                                        setResult(data);
                                        setIsComplete(true);
                                        setIsStreaming(false);
                                        if (options.onComplete) {
                                            options.onComplete(data);
                                        }
                                    } else if (data.event === 'error') {
                                        setEvents(prev => [...prev, { message: data.message, type: 'error' }]);
                                        setError(data.message);
                                        setIsStreaming(false);
                                        if (options.onError) {
                                            options.onError(data.message);
                                        }
                                    } else if (data.event === 'start') {
                                        setEvents(prev => [...prev, { message: data.message, type: 'info' }]);
                                    }
                                } catch (e) {
                                    console.error('Error parsing SSE data:', e);
                                }
                            }
                        });

                        readStream();
                    });
                };

                readStream();
            })
            .catch(err => {
                setError(err.message);
                setIsStreaming(false);
                if (options.onError) {
                    options.onError(err.message);
                }
            });
    };

    const stopStreaming = () => {
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }
        setIsStreaming(false);
    };

    useEffect(() => {
        return () => {
            stopStreaming();
        };
    }, []);

    return {
        events,
        isStreaming,
        isComplete,
        error,
        result,
        startStreaming,
        stopStreaming,
    };
};

export default StreamingDisplay;
