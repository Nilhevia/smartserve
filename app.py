from function import *
# from speech_to_text_function import *
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, static_url_path='/static')
@app.route("/")
def beranda():
    return render_template('home.html')

@app.route("/api/deteksi", methods=['POST'])
def apiDeteksi():
    # Nilai default untuk string input
    text_input = ""
    if request.method == 'POST':
        # Set nilai string input dari pengguna
        text_input = request.form['data']
        # Membuat Input Chat
        texts_p = []
        prediction_input = text_input
        prediction_input = convert_slang_to_normal(prediction_input)
        # Menghapus punktuasi dan konversi ke huruf kecil
        prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
        prediction_input = ''.join(prediction_input)
        texts_p.append(prediction_input)
        # Tokenisasi dan Padding
        prediction_input = tokenizer.texts_to_sequences(texts_p)
        prediction_input = np.array(prediction_input).reshape(-1)
        prediction_input = pad_sequences([prediction_input],input_shape)
        # Mendapatkan hasil keluaran pada model
        output = model.predict(prediction_input)
        output = output.argmax()        

        # Menemukan respon sesuai data tag dan memainkan voice bot
        response_tag = le.inverse_transform([output])[0]
        tts = gTTS(random.choice(responses[response_tag]), lang='id')
        # Simpan model voice bot ke dalam Google Drive
        tts.save('SmartServe.wav')
        if response_tag == "makanan":
            return jsonify({
            "data": responses[response_tag],
            "type": response_tag
            })
        elif response_tag == "minuman":
            return jsonify({
            "data": responses[response_tag],
            "type": response_tag
            })
        elif response_tag == "sidedish":
            return jsonify({
            "data": responses[response_tag],
            "type": response_tag
            })
        else:
            return jsonify({
            "data": random.choice(responses[response_tag]),
            "type": response_tag
            })
        
@app.route("/api/speech", methods=['POST'])
def apiSpeech():
    #record
    filename = 'voice.wav'
    if request.method == 'POST':
        record(filename, 5)
        return true

# =[Main]========================================

if __name__ == '__main__':

    # Setup

    # Run Flask di localhost
    app.run(host="localhost", port=5000, debug=True)
