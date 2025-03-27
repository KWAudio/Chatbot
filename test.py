from transformers import PreTrainedTokenizerFast
import torch
from transformers import GPT2LMHeadModel

# 토크나이저와 모델 불러오기
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

# 대화 시작
print("손자와의 대화를 시작합니다! (종료하려면 '그만'을 입력하세요.)")

# 대화 흐름 유지
conversation_history = ""  # 이전 대화를 저장할 변수

while True:
    # 할아버지의 입력 받기
    grandparent_input = input("할아버지: ")

    # "그만" 입력 시 종료
    if grandparent_input.strip().lower() == '그만':
        print("손자와의 대화를 종료합니다. 안녕히 가세요!")
        break

    # 할아버지의 말에 손자가 대답하도록 하기 위해, 이전 대화와 이어지도록 설정
    conversation_history += f"할아버지: {grandparent_input}\n"

    # 모델에 입력 (대화 역사 포함)
    input_ids = tokenizer.encode(conversation_history, return_tensors='pt')

    # 문장 생성
    gen_ids = model.generate(input_ids,
                             max_length=100,  # 적당히 제한된 길이로 설정
                             repetition_penalty=1.5,  # 반복문제 방지
                             top_p=0.92,  # 텍스트 다양성 증가
                             temperature=0.7,  # 예측 가능하게
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             bos_token_id=tokenizer.bos_token_id,
                             use_cache=True)

    # 생성된 문장 디코딩
    generated = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    # 손자의 답변 출력
    handson_text = generated.split("할아버지:")[-1].strip()  # 할아버지의 입력 제외
    conversation_history += f"손자: {handson_text}\n"  # 대화 기록에 손자 말 추가
    print(f"손자: {handson_text}")
