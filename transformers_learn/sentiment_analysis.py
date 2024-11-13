from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="SamLowe/roberta-base-go_emotions", device="cpu")

resp = classifier("i'm so hungry, i want to eat something")
print("resp", resp)

mutil_resp = classifier(["i'm so hungry, i want to eat something", "i'm so happy"])
print("mutil_resp", mutil_resp)
