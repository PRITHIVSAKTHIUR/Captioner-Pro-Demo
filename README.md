# Captioner-Pro-Demo

Caption-Anything-InContext is a dataset curated using the model [Caption-Pro](https://huggingface.co/prithivMLmods/Caption-Pro) for improved in-context captioning of images. This model is designed for generating multiple captions for images, ensuring they are contextually accurate.

### Required Lib
```py
!pip install -q transformers qwen-vl-utils==0.0.2
```

Demo with transformers

```py
import os
import gdown
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO

# Define the Google Drive folder ID and local download directory
GDRIVE_FOLDER_ID = "1hMZyonEVLLRDHOy4lnGQFgB5EuL3pnxq"
DOWNLOAD_DIR = "downloaded_images"

# Ensure the download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# 1. Load the model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "prithivMLmods/JSONify-Flux",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("prithivMLmods/Caption-Pro")

def download_images():
    """Download images from a Google Drive folder using gdown."""
    print("Downloading images from Google Drive...")
    gdown.download_folder(id=GDRIVE_FOLDER_ID, output=DOWNLOAD_DIR, quiet=False)

def encode_image_to_base64(image):
    """Encode a PIL image to base64 (for storing directly in a Parquet file)."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def process_and_display_images():
    data = []

    # 2. Loop through downloaded images
    for filename in os.listdir(DOWNLOAD_DIR):
        image_path = os.path.join(DOWNLOAD_DIR, filename)
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        try:
            # 2a. Open the image with PIL
            image = Image.open(image_path).convert("RGB")

            # 2b. Create the messages with the *actual* image
            #    (Minimal textual instruction, but you can customize.)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Generate a detailed and optimized caption for the given image."},
                    ],
                }
            ]

            # 3. Prepare the inputs for Qwen-VL
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

            # 4. Generate the caption
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            # Remove the portion of the output that duplicates input tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            # 5. Show the image + caption
            plt.figure()
            plt.imshow(image)
            plt.axis("off")
            plt.figtext(
                0.5, 0.01,
                f"Caption: {output_text}",
                wrap=True,
                horizontalalignment='center',
                fontsize=12,
                color='black'
            )
            plt.show()

            # 6. Store results (image in base64 + generated caption)
            image_base64 = encode_image_to_base64(image)
            data.append({"image": image_base64, "caption": output_text})

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # 7. Convert the list of dicts to a DataFrame and save as Parquet
    df = pd.DataFrame(data)
    df.to_parquet("image_captions_dataset.parquet", index=False)
    print("Dataset saved as image_captions_dataset.parquet")

# Run the pipeline
download_images()
process_and_display_images()
```

```python
/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
config.json: 100%
 1.25k/1.25k [00:00<00:00, 99.8kB/s]
model.safetensors: 100%
 4.42G/4.42G [01:45<00:00, 41.7MB/s]
`Qwen2VLRotaryEmbedding` can now be fully parameterized by passing the model config through the `config` argument. All other arguments will be removed in v4.46
generation_config.json: 100%
 252/252 [00:00<00:00, 18.1kB/s]
preprocessor_config.json: 100%
 596/596 [00:00<00:00, 41.4kB/s]
tokenizer_config.json: 100%
 4.47k/4.47k [00:00<00:00, 298kB/s]
vocab.json: 100%
 2.78M/2.78M [00:00<00:00, 9.62MB/s]
merges.txt: 100%
 1.82M/1.82M [00:00<00:00, 36.2MB/s]
tokenizer.json: 100%
 11.4M/11.4M [00:00<00:00, 36.0MB/s]
added_tokens.json: 100%
 408/408 [00:00<00:00, 26.9kB/s]
special_tokens_map.json: 100%
 645/645 [00:00<00:00, 46.1kB/s]
chat_template.json: 100%
 1.05k/1.05k [00:00<00:00, 71.0kB/s]
Downloading images from Google Drive...
Retrieving folder contents
Processing file 1keTGdE06rGOPl0rR8vFyymrc0ISZM__p 00000.jpg
Processing file 14vKlJaVjCXJ8htEL4qeV-at3M4vjD7j- 00001.jpg
Processing file 1DG-Es0eIvor4nyonr2rHjtZO6-kCRkCe 00002.jpg
Processing file 1yQ98PuIcSxd6nmHdsDxYKNC0gFV5axYV 00003.jpg
Processing file 132BOr0rFYEbYeG9NzyQwtZdL4gBdR9lt 00004.jpg
Processing file 1l0bdGptC2ykGarqQBMhYAlARIN2ITEiG 00005.jpg
Processing file 1aA87FcjaOKio9jqSStssPiCrbpRUh1Qq 00006.jpg
Processing file 189hlEjG8F-55F2FfBdjccVzYD-N-lM8V 00007.jpg
Processing file 1Ac4FzLEPazfkizFbybaTAQ-6v9_1gBEm 00009.jpg
Processing file 1QblETVnyLLr3UndjhLZQQpHLUW2U9FIf 00010.jpg
Retrieving folder contents completed
Building directory structure
Building directory structure completed
Downloading...
From: https://drive.google.com/uc?id=1keTGdE06rGOPl0rR8vFyymrc0ISZM__p
To: /content/downloaded_images/00000.jpg
100%|██████████| 9.15k/9.15k [00:00<00:00, 22.8MB/s]
Downloading...
From: https://drive.google.com/uc?id=14vKlJaVjCXJ8htEL4qeV-at3M4vjD7j-
To: /content/downloaded_images/00001.jpg
100%|██████████| 8.47k/8.47k [00:00<00:00, 6.43MB/s]
Downloading...
From: https://drive.google.com/uc?id=1DG-Es0eIvor4nyonr2rHjtZO6-kCRkCe
To: /content/downloaded_images/00002.jpg
100%|██████████| 8.73k/8.73k [00:00<00:00, 16.8MB/s]
Downloading...
From: https://drive.google.com/uc?id=1yQ98PuIcSxd6nmHdsDxYKNC0gFV5axYV
To: /content/downloaded_images/00003.jpg
100%|██████████| 13.0k/13.0k [00:00<00:00, 24.9MB/s]
Downloading...
From: https://drive.google.com/uc?id=132BOr0rFYEbYeG9NzyQwtZdL4gBdR9lt
To: /content/downloaded_images/00004.jpg
100%|██████████| 10.2k/10.2k [00:00<00:00, 27.0MB/s]
Downloading...
From: https://drive.google.com/uc?id=1l0bdGptC2ykGarqQBMhYAlARIN2ITEiG
To: /content/downloaded_images/00005.jpg
100%|██████████| 8.17k/8.17k [00:00<00:00, 25.2MB/s]
Downloading...
From: https://drive.google.com/uc?id=1aA87FcjaOKio9jqSStssPiCrbpRUh1Qq
To: /content/downloaded_images/00006.jpg
100%|██████████| 10.9k/10.9k [00:00<00:00, 35.3MB/s]
Downloading...
From: https://drive.google.com/uc?id=189hlEjG8F-55F2FfBdjccVzYD-N-lM8V
To: /content/downloaded_images/00007.jpg
100%|██████████| 8.74k/8.74k [00:00<00:00, 23.5MB/s]
Downloading...
From: https://drive.google.com/uc?id=1Ac4FzLEPazfkizFbybaTAQ-6v9_1gBEm
To: /content/downloaded_images/00009.jpg
100%|██████████| 10.2k/10.2k [00:00<00:00, 30.1MB/s]
Downloading...
From: https://drive.google.com/uc?id=1QblETVnyLLr3UndjhLZQQpHLUW2U9FIf
To: /content/downloaded_images/00010.jpg
100%|██████████| 9.34k/9.34k [00:00<00:00, 25.6MB/s]
Download completed

Dataset saved as image_captions_dataset.parquet
```

![download (1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/tqBzqnq9of0DwUJx1Jaur.png)
![download (2).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/JBv1kAVXqQHMrzmac4AkC.png)
![download (3).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/LP3D7mR5k9tRLp9Tjl9O9.png)
![download (4).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/83KCJojE98PR6k5yu6LuL.png)
![download (5).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/ZJKxcz3SE1-lkc04aWVut.png)
![download (6).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/JahVRiRza9wBthbrhiOj4.png)
![download (7).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/W1oZbjSd2Gri7MRb5CnE8.png)
![download (8).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/QTtDGt2s20lmHn6bsqAHm.png)
![download (9).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/-3mEzMxLTA3J_Zo1cYTfj.png)
![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/gLDsD7nDknYx0APx3o3IN.png)
