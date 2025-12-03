from transformers import DistilBertTokenizer, DistilBertModel

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
text = "some text for model"
encoded_input = tokenizer(text, return_tensors = 'pt')
output = model(**encoded_input)
output

output.last_hidden_state.shape

#Autotokensizer and Automodel
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
text = "some text for model"
encoded_input = tokenizer(text, return_tensors = 'pt')
output = model(**encoded_input)
output


#classification head with AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
text = "I love using transformers library!"
encoded_input = tokenizer(text, return_tensors = 'pt')
output = model(**encoded_input)
output

#postprocessing
import numpy as np               
logits = output.logits.detach().numpy()
y_pred = np.argmax(logits)
y_pred

#normalize 
print(output.logits.softmax(dim=-1).tolist())

#prediction on gpu
device = 'cpu'
model = model.to(device)
text = "I enjoyed watching the movie!"
encoded_input = tokenizer(text, return_tensors = 'pt').to(device)
output = model(**encoded_input)
logits = output.logits.detach().cpu().numpy()
y_pred = np.argmax(logits)
y_pred


#predicting multiple text
device = 'cpu'
model = model.to(device)
texts = ["I enjoyed watching the movie!", "The film was boring and too long."]
encoded_input = tokenizer(texts, return_tensors = 'pt', padding = True, truncation = True).to(device) 
output = model(**encoded_input)
logits = output.logits.detach().cpu().numpy()
y_pred = np.argmax(logits, axis = -1)
y_pred




from transformers import AutoTokenizer, AutoModelForTokenClassification
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased")
text = "Hugging Face is creating a tool that the community uses to solve NLP tasks."
encoded_input = tokenizer(text, return_tensors = 'pt')
output = model(**encoded_input)
output

import numpy as np

logits = output.logits.detach().numpy()
y_pred = np.argmax(logits, axis = -1)
y_pred

logits.shape

device = 'cpu'
model = model.to(device)
text = 'I enjoyed watching a movie!'
encoded_input = tokenizer(text, return_tensors = 'pt').to(device)
output = model(**encoded_input)
logits = output.logits.detach().cpu().numpy()
y_pred = np.argmax(logits, axis = -1)
y_pred


from transformers import pipeline
classifier = pipeline("zero-shot-classification", model = "facebook/bart-large-mnli")

sequence_to_classify = "One day I will see the world"
candidate_labels = ["travel", "cooking", "dancing"]

classifier(sequence_to_classify, candidate_labels)

