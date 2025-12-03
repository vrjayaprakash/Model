from transformers import pipeline

classifier = pipeline("zero-shot-classification", model = "facebook/bart-large-mnli")

sequence_to_classify = "Who are you voting for in 2024?"
candidate_labels = ["politics", "public health", "economy", "education"]

classifier(sequence_to_classify, candidate_labels)

candidate_labels = ["politics", "public health", "economy", "education", "sports"]
classifier(sequence_to_classify, candidate_labels, multi_labels = True)

sequence_to_classify = "Donald trump will be the next president"
candidate_labels = ["science", "politics", "history"]
classifier(sequence_to_classify, candidate_labels)

from transformers import pipeline

classifier = pipeline("zero-shot-classification", model = "joeddav/distilbert-base-uncased-go-emotions-student") 

sequence  = " I am so happy and joyful!"  
label = ["joy", "sadness", "anger", "fear", "sad" ]    

classifier(sequence, label, multi_labels = True)


from transformers import pipeline
text_generator = pipeline("text-generation", model = "gpt2")
text = "I want to download an"

text_generator(text, max_length = 30)


from transformers import pipeline
ner = pipeline("ner", model = "dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

sentence = "Sundar pichai is the ceo of Google"
ner(sentence)

from transformers import pipeline
summarizer = pipeline("summarization", model = "facebook/bart-large-cnn")

text = """
I love to visit new places and explore different cultures. 
Traveling allows me to experience the world in a unique way, meeting new people and trying new foods. 
One of my favorite destinations is Japan, where the blend of traditional and modern culture creates a fascinating atmosphere. 
From the bustling streets of Tokyo to the serene temples of Kyoto, there is always something new to discover.
Traveling not only broadens my horizons but also helps me appreciate the diversity of our world.
"""

summarizer(text, max_length = 50, min_length = 25)









