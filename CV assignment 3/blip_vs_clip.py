import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
import os
import matplotlib.pyplot as plt
from open_clip import create_model_from_pretrained, get_tokenizer
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


blip_cap_processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
blip_cap_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to(device)


individual_image_path = r"C:\Users\Asus\Documents\CV assignment 3\sample_image.jpg"
single_img = Image.open(individual_image_path).convert('RGB')

inputs = blip_cap_processor(single_img, return_tensors='pt').to(device)
with torch.no_grad():
    out = blip_cap_model.generate(**inputs, max_new_tokens=50)
single_desc = blip_cap_processor.decode(out[0], skip_special_tokens=True)

print(f"Individual Image Description: {single_desc}")

sample_folder_path = r"C:\Users\Asus\Documents\CV assignment 3\samples"
images = []
descriptions = []

images.append(single_img)
descriptions.append(f"MAIN IMAGE: {single_desc}")

valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
for filename in os.listdir(sample_folder_path):
    if filename.lower().endswith(valid_extensions):
        full_path = os.path.join(sample_folder_path, filename)
        img = Image.open(full_path).convert('RGB')
        
        inputs = blip_cap_processor(img, return_tensors='pt').to(device)
        with torch.no_grad():
            out = blip_cap_model.generate(**inputs, max_new_tokens=50)
        
        desc = blip_cap_processor.decode(out[0], skip_special_tokens=True)
        images.append(img)
        descriptions.append(desc)

num_total = len(images)
cols = 3
rows = (num_total // cols) + (num_total % cols > 0)

plt.figure(figsize=(15, 5 * rows))
for i, (img, desc) in enumerate(zip(images, descriptions)):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.title(desc, fontsize=9, wrap=True)
    plt.axis('off')

plt.tight_layout()
plt.show()

clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

plt.figure(figsize=(15, 5 * rows))

for i, (img, desc) in enumerate(zip(images, descriptions)):
    clip_inputs = processor(text=descriptions, images=img, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        output = clip_model(**clip_inputs)
    probs  = output.logits_per_image.softmax(dim = -1)
    score = probs[0][i].item()
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.title(f"BLIP: {desc}\nCLIP Logit: {score:.2f}", fontsize=9, wrap=True)
    plt.axis('off')

plt.tight_layout()
plt.show()


OPEN_CLIP_MODEL = 'hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-Recap-DataComp-1B' 

open_clip_model, preprocess = create_model_from_pretrained(OPEN_CLIP_MODEL)
open_clip_model = open_clip_model.to(device).eval()
tokenizer = get_tokenizer(OPEN_CLIP_MODEL)

plt.figure(figsize=(15, 5 * rows))
for i, (img, desc) in enumerate(zip(images, descriptions)):
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    all_text_tokens = tokenizer(descriptions).to(device)
    with torch.no_grad():
        image_features = open_clip_model.encode_image(img_tensor)
        all_text_features = open_clip_model.encode_text(all_text_tokens)
    image_features = F.normalize(image_features, dim=-1)
    all_text_features = F.normalize(all_text_features, dim=-1)
    probs = (100.0 * image_features @ all_text_features.T).softmax(dim=-1)
    own_score = probs[0][i].item()

    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.title(f"BLIP: {desc}\nCLIPS Score: {own_score:.4f}", fontsize=9, wrap=True)
    plt.axis('off')

plt.tight_layout()
plt.show()




