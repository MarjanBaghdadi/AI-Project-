import requests
import os
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
from PIL import Image

root_dir = "Dataset/ImageNet_Dataset/"
image_urls = Path(root_dir + "Selected_Class_URLS")
save_dir = Path(root_dir + "Downloaded_images")
image_net_ref = pd.DataFrame(columns=['Image_Name', 'Source_URL', 'Class_Name'])
desired_image_per_class = 115

if not save_dir.exists():
    save_dir.mkdir()

for classes in list(image_urls.iterdir()):
    for url_file in classes.iterdir():
        print("Current Class: ", url_file)
        with open(url_file, 'r', encoding='utf-8', errors='ignore') as f:
            class_urls = f.readlines()
            class_image_count = 0
            last_successful_attempt = 0

            for index, url in enumerate(class_urls[-100:]):
                url_parsed_data = urlparse(url)
                image_name = os.path.basename(url_parsed_data.path).replace("\n", "")

                if last_successful_attempt > 50:
                    break

                try:
                    image_data = requests.get(url, timeout=5).content
                    with open(os.path.join(save_dir.absolute(), image_name), 'wb') as file_handler:
                        file_handler.write(image_data)

                    try:
                        im = Image.open(os.path.join(save_dir.absolute(), image_name))
                    except IOError:
                        os.remove(os.path.join(save_dir.absolute(), image_name))
                        last_successful_attempt += 1
                        print(f"url index: {index}, {last_successful_attempt}")
                        continue

                    image_net_ref = image_net_ref.append({
                        'Image_Name': image_name,
                        'Source_URL': url.replace("\n", ""),
                        'Class_Name': url_file.name.replace('.txt', '')
                    }, ignore_index=True)

                    class_image_count += 1
                    last_successful_attempt = 0
                    if class_image_count >= desired_image_per_class:
                        break
                except:
                    continue

image_net_ref.to_csv(root_dir + "ImageNet_References.csv", index=False)
