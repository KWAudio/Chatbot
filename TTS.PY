import pyttsx3

def speak_with_child_voice(text):
    engine = pyttsx3.init()

    # 모든 음성 목록 출력
    voices = engine.getProperty('voices')

    # 목소리 설정 
    engine.setProperty('voice', voices[0].id)  

    # 속도 설정 (느리게 설정 가능)
    engine.setProperty('rate', 150)  # 속도 낮추기 (기본 200)

    # 볼륨 설정 (기본값은 1.0)
    engine.setProperty('volume', 1.0)

    # 텍스트를 음성으로 출력
    engine.say(text)
    engine.runAndWait()

# 예시로 문장 생성
speak_with_child_voice("할아버지, 나 오늘 정말 재미있는 일이 있었어요!")
