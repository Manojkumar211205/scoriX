import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const apiService = {
    // Generate question paper from text
    generateFromText: async (text, collectionName = 'test_ai_collection_v1') => {
        try {
            const response = await api.post('/api/generate', {
                text,
                collection_name: collectionName,
            });
            return { success: true, data: response.data };
        } catch (error) {
            return {
                success: false,
                error: error.response?.data?.message || error.message || 'Failed to generate question paper',
            };
        }
    },

    // Generate question paper from file
    generateFromFile: async (file, collectionName = 'test_ai_collection_v1') => {
        try {
            const formData = new FormData();
            formData.append('file', file);
            if (collectionName) {
                formData.append('collection_name', collectionName);
            }

            const response = await api.post('/api/generate-from-file', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            return { success: true, data: response.data };
        } catch (error) {
            return {
                success: false,
                error: error.response?.data?.message || error.message || 'Failed to upload and generate',
            };
        }
    },

    // Get list of all outputs
    getOutputs: async () => {
        try {
            const response = await api.get('/api/outputs');
            return { success: true, data: response.data };
        } catch (error) {
            return {
                success: false,
                error: error.response?.data?.message || error.message || 'Failed to fetch outputs',
            };
        }
    },

    // Download output file
    downloadOutput: (filename) => {
        return `${API_BASE_URL}/api/outputs/${filename}`;
    },

    // Health check
    healthCheck: async () => {
        try {
            const response = await api.get('/health');
            return { success: true, data: response.data };
        } catch (error) {
            return { success: false, error: 'API server is not responding' };
        }
    },
};

export default apiService;
