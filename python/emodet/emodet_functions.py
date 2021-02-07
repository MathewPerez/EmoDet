 ### Imports ###
import string 
import torch
import numpy as np
import logging
logging.getLogger().setLevel(logging.INFO)
from transformers import BertForSequenceClassification, BertTokenizer

### Initialize trained BERT model ###
model_dir = "./emod_model_save"
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)

# If there's a GPU available...
if torch.cuda.is_available():    

	# Tell PyTorch to use the GPU.    
	device = torch.device("cuda")

	logging.info('There are %d GPU(s) available.' % torch.cuda.device_count())

	logging.info('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
	logging.info('No GPU available, using the CPU instead.')
	device = torch.device("cpu")

### Define the labels map ###
labels = [
		'admiration',
		'amusement',
		'anger',
		'annoyance',
		'approval',
		'caring',
		'confusion',
		'curiosity',
		'desire',
		'disappointment',
		'disapproval',
		'disgust',
		'embarrassment',
		'excitement',
		'fear',
		'gratitude',
		'grief',
		'joy',
		'love',
		'nervousness',
		'optimism',
		'pride',
		'realization',
		'relief',
		'remorse',
		'sadness',
		'surprise',
		'neutral'  
	]
label_dict = { i:v for i,v in enumerate(labels)}

### Define base prediction class ###
class EmotionPred():	
	### Prediction function to use ###
	@staticmethod
	def predict(text, nop=True, length=100, thresh=0.3, n=0):
		# Move model to gpu/cpu depending on environment
		model.to(device)
		n = int(n)
		if (nop):
			# Strip punctuation
			text = ' '.join(word.strip(string.punctuation) for word in text.split())
		# First encode the text with the model's tokenizer
		encoded_dict = tokenizer.encode_plus(
												text,                      # Sentence to encode.
												add_special_tokens = True, # Add '[CLS]' and '[SEP]'
												max_length = length,           # Pad & truncate all sentences.
												padding = True,
												return_attention_mask = True,   # Construct attn. masks.
												return_tensors = 'pt',     # Return pytorch tensors.
								)
		
		input_id = encoded_dict['input_ids']
		attention_mask = encoded_dict['attention_mask']
		
		# Put model in evaluation mode
		model.eval()

		# Produce logits with forward pass and no computation graph
		with torch.no_grad():
			# Forward pass, calculate logit predictions
			outputs = model(input_id, token_type_ids=None, 
					attention_mask=attention_mask)

		# Get probabilities from logits
		logits = outputs[0]
		probs = torch.sigmoid(logits).cpu().detach().numpy().tolist()[0]

		# If given n emotions to produce
		if (n>0):
			# Get predicted classes
			final_preds = sorted(range(len(probs)), key=lambda i: probs[i])[-n:]
			predicted_labels = [label_dict[c] for c in final_preds]
		# Use threshold instead
		else:
			# Get predicted classes
			final_preds = [i for i,val in enumerate(probs) if val>=thresh]
			predicted_labels = [label_dict[c] for c in final_preds]
		return predicted_labels
