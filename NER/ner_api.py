from flask import Flask, request, jsonify
from flask_cors import CORS
from Ner_Bert import NE_Extraction

app = Flask(__name__)
CORS(app)
NER_labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
tag2idx = {t: i for i, t in enumerate(NER_labels)}
ner_extractor = NE_Extraction(NER_labels, tag2idx)  # Đổi tên biến ở đây

@app.route('/api/ner', methods=['POST'])
def process_ner():  # Đổi tên hàm ở đây
    data = request.get_json()
    text = data['text']
    print(text)
    text_mark = ner_extractor.extract(text) 
    print(text_mark)
    return jsonify({
        'text': text_mark
    })

if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')
