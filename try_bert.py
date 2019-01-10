import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# Predict all tokens
predictions = model(tokens_tensor, segments_tensors)

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
assert predicted_token == 'henson'


