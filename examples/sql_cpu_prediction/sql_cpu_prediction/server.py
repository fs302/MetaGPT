## server.py
from flask import Flask, request, jsonify
try:
    from preprocessor import Preprocessor
    from model import Model
except ImportError as e:
    print(f"Failed to import modules: {e}")
    exit(1)
import config
from waitress import serve

class Server:
    def __init__(self, preprocessor=None, model=None):
        self.app = Flask(__name__)
        self.preprocessor = preprocessor if preprocessor else Preprocessor()
        self.model = model if model else Model()
        self._initialize_routes()

    def _initialize_routes(self):
        @self.app.route('/predict', methods=['POST'])
        def predict_cpu_cost():
            return self._handle_predict_cpu_cost()

        @self.app.route('/classify', methods=['POST'])
        def classify_query():
            return self._handle_classify_query()

    def start(self, host='127.0.0.1', port=8080):
        serve(self.app, host=host, port=port)

    def _handle_predict_cpu_cost(self):
        sql_query = request.json.get('sql_query', '')
        if not sql_query:
            return jsonify({'error': 'No SQL query provided'}), 400
        try:
            preprocessed_data = self.preprocessor.preprocess_sql(sql_query)
            prediction = self.model.predict(preprocessed_data)
            return jsonify({'prediction': prediction.tolist()}), 200
        except Exception as e:
            return jsonify({'error': f'Failed to predict CPU cost due to: {str(e)}'}), 500

    def _handle_classify_query(self):
        sql_query = request.json.get('sql_query', '')
        if not sql_query:
            return jsonify({'error': 'No SQL query provided'}), 400
        try:
            preprocessed_data = self.preprocessor.preprocess_sql(sql_query)
            classification = self.model.classify(preprocessed_data)
            return jsonify({'classification': classification.tolist()}), 200
        except Exception as e:
            return jsonify({'error': f'Failed to classify SQL query due to: {str(e)}'}), 500

if __name__ == '__main__':
    server = Server()
    server.start()
