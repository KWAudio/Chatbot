from transformers import AutoModel, AutoTokenizer

# tokenizer & model 로드 (KoBERT or 다른 BERT 기반 모델)
model = AutoModel.from_pretrained("monologg/kobert")
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

# 입력 문장
text = "오늘 길을 걷다가 넘어졌어"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# 예측
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs)

labels = ["부정", "중립", "긍정"]
print(f"예측 감정: {labels[pred]}")
