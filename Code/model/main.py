from transformers import BertTokenizer, TFBertForSequenceClassification
from modulos.model import create_bert

import modulos.model as model
model, tokenizer = create_bert(model_name='bert-base-uncased')
text = "hola, como estás"
inputs = tokenizer(text, return_tensors='tf', padding="max_length", truncation=True, max_length=256)

outputs = model(inputs)
logits = outputs[0]
import tensorflow as tf
probabilities = tf.nn.softmax(logits, axis=-1)

print(probabilities)
