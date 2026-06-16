#pip install open_clip_torch

import torch
from PIL import Image
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer

model, preprocess = create_model_from_pretrained('hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-Recap-DataComp-1B')
tokenizer = get_tokenizer('hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-224-Recap-DataComp-1B')
image = Image.open(r"C:\Users\Asus\Documents\CV assignment 3\sample_image.jpg")
image = preprocess(image).unsqueeze(0)



text = tokenizer([
    "A man holds a large Dog.",
    "He wears a white shirt.",
    "His trousers are a light grey.",
    "The dog has grey fur.",
    "A white patch marks the dog's chest.",
    "They stand on a wooden floor.",
    "full bookshelf sits behind them.",
    "The man wears brown leather shoes.",
    "A white door is in the background.",
    "The dog looks very relaxed."], context_length=model.context_length)


with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)


print("Label probs:", text_probs) 