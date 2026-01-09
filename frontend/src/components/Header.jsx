import { motion } from 'framer-motion';
import { GraduationCap, Sparkles } from 'lucide-react';

const Header = () => {
    return (
        <motion.header
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="header"
        >
            <div className="header-content">
                <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
                    className="icon-wrapper"
                >
                    <GraduationCap size={48} className="header-icon" />
                    <Sparkles size={24} className="sparkle-icon" />
                </motion.div>
                <h1 className="gradient-text">ScoriX Question Paper Generator</h1>
                <p className="subtitle">AI-Powered Question Paper Generation</p>
            </div>
        </motion.header>
    );
};

export default Header;
