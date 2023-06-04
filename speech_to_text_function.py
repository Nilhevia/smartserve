# all imports
import speech_recognition as sr
from IPython.display import Javascript
from google.colab import output
from base64 import b64decode

RECORD = """
const sleep = time => new Promise(resolve => {
setTimeout(resolve, time)
}, )
const b2text = blob => new Promise(resolve => {
const reader = new FileReader()
reader.onloadend = e => resolve(e.srcElement.result)
reader.readAsDataURL(blob)
})
var espacio = document.querySelector("#output-area")
var record = time => new Promise(async resolve => {
stream = await navigator.mediaDevices.getUserMedia({ audio: true })
recorder = new MediaRecorder(stream)
chunks = []
recorder.ondataavailable = e => chunks.push(e.data)
recorder.start()
var numerillo = (time/1000)-1
for (var i = 0; i < numerillo; i++) {
espacio.appendChild(document.createTextNode(numerillo-i))
await sleep(1000)
espacio.removeChild(espacio.lastChild)
}
recorder.onstop = async ()=>{
blob = new Blob(chunks)
text = await b2text(blob)
resolve(text)
}
recorder.stop()
})
"""

def record(fn, sec):
  display(Javascript(RECORD))
  s = output.eval_js('record(%d)' % (sec*1000))
  b = b64decode(s.split(',')[1])
  with open(fn,'wb') as f:
    f.write(b)
  return fn