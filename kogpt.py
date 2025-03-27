from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')

import torch
from transformers import GPT2LMHeadModel

# 모델 로드
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

# 할아버지, 할머니에게 대답하는 문장 스타일을 유도하는 프롬프트
text = '할아버지, 요즘 날씨가 좋아요'

input_ids = tokenizer.encode(text, return_tensors='pt')

# 생성 길이 제한을 줄여서 짧고 간단한 문장을 출력하도록 설정
gen_ids = model.generate(input_ids,
                           max_length=60,  # 짧은 문장으로 설정
                           repetition_penalty=1.5,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           bos_token_id=tokenizer.bos_token_id,
                           use_cache=True)

generated = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

# 출력 결과 확인
print(generated)
