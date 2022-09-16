from email.mime import image
from transformers import CLIPModel,CLIPProcessor
from PIL import Image
import requests

model = CLIPModel.from_pretrained('./data')
processor = CLIPProcessor.from_pretrained('./data')

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim = 1)

print(probs)

