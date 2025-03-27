import speech_recognition as sr
from gtts import gTTS
import os
import playsound
import time
from datetime import datetime

def speak(text):
    tts = gTTS(text=text, lang='ko')   # 텍스트를 음성으로 변환하여 출력
    filename = 'voice.mp3'
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

def get_audio():
    # 음성 인식
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = " "
        try:
            said = r.recognize_google(audio, language="ko-KR")   #Google API(recognize_google)를 사용하여 음성을 텍스트로 변환
        except Exception as e:
            pass
        return said

# 프로그램 시작
if os.path.isfile('memo.txt'):
    os.remove('memo.txt')

# 음성 안내
speak("안녕하세요. 2초 후에 말씀하시고, 종료시 '굿바이'라고 말씀하시면 됩니다.")

while True:
    text = get_audio()  # 음성 인식
    print(text)

    # 텍스트 파일에 저장
    with open('memo.txt', 'a') as f:
        f.write(str(text) + "\n")

    # "굿바이"가 포함된 경우 종료
    if "굿바이" in text:
        break

    time.sleep(0.1)
