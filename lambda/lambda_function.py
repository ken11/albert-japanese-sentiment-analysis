from transformers import AlbertTokenizerFast, AlbertForSequenceClassification
import numpy as np
import json


def inference(event, context):
    model = AlbertForSequenceClassification.from_pretrained("./model")
    tokenizer = AlbertTokenizerFast.from_pretrained("./model")

    inputs = tokenizer(event["body"], return_tensors="pt", padding='max_length', truncation=True, max_length=128)
    pred = model(**inputs).logits[0]
    pred = np.argmax(pred.detach().numpy(), axis=-1)
    print(f"input text: {event['body']}\nsentiment: {model.config.id2label[pred]}")

    return {
        'isBase64Encoded': False,
        'statusCode': 200,
        'headers': {},
        'body': json.dumps({"input text": event['body'], "sentiment": model.config.id2label[pred]})
    }
