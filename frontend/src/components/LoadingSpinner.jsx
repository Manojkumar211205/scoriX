import { motion } from 'framer-motion';

const LoadingSpinner = ({ message = 'Processing...' }) => {
    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="loading-container"
        >
            <div className="spinner"></div>
            <p className="loading-message">{message}</p>
        </motion.div>
    );
};

export default LoadingSpinner;
