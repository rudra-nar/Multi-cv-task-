import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_ID = 'Salesforce/blip-vqa-base'
processor = BlipProcessor.from_pretrained(MODEL_ID)
model     = BlipForQuestionAnswering.from_pretrained(MODEL_ID).to(device)

image = Image.open(r"C:\Users\Asus\Documents\CV assignment 3\sample_image.jpg")
r"C:\Users\Asus\Documents\CV assignment 3\sample_image.jpg"

questions = ['Where is the dog present in the image?','Can you tell where the dog is located in the picture?',
             'In which part of the image is the dog positioned?','Where is the dog relative to the man in the image?',
             '“Describe the location of the dog using context.','What is the dog doing and where is it in the scene?']

for question in questions:  
    inputs = processor(image, question,return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)  # fix 1 & 2
    answer = processor.decode(output[0], skip_special_tokens=True)
    print(f'Q: {question}')
    print(f'A: {answer}')
    print()

man ='Where is the man present in the image?'
inputs = processor(image, man,return_tensors='pt').to(device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)  # fix 1 & 2
answer = processor.decode(output[0], skip_special_tokens=True)
print(f'Q: {man}')
print(f'A: {answer}')
print()

# the model fails in answering question which are relative to othewr objects, 
#model works fine for descreptive answers with clear anchors, where cross attention allings text
#but it fails where question allow for a shortcut ie can you tell me where the dog is?, it uses 
#languigage prior.


questions.append(man)

embeddings = []
for q in questions:
    inputs = processor(image, q, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            output_hidden_states=True, 
            return_dict_in_generate=True, 
            max_new_tokens=1
        )

    last_hidden = out.hidden_states[0][-1]

    embeddings.append(last_hidden.mean(dim=1).cpu().numpy().flatten())

reduced = TSNE(n_components=2, perplexity=min(2, len(questions)-1), random_state=42).fit_transform(np.array(embeddings))

plt.figure(figsize=(12, 7))
plt.scatter(reduced[:, 0], reduced[:, 1], c='blue', s=100, edgecolors='black')

for i, txt in enumerate(questions):
    plt.annotate(
        txt, 
        (reduced[i, 0], reduced[i, 1]), 
        xytext=(5, 5), 
        textcoords='offset points',
        fontsize=9
    )

plt.title('t-SNE of BLIP Decoder Embeddings for Different Questions')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()