import pandas as pd
from pathlib import Path

dataset_dir = Path("Dataset/self-built-masked-face-recognition-dataset")
mask_dir = dataset_dir/"AFDB_masked_face_dataset"
mask_less_dir = dataset_dir/"AFDB_face_dataset"

# creating a dataframe object to store the masked and no masked images with their labels
df_object = pd.DataFrame()


# label all the images with mask to "1"
# there are 520 subjects with mask. (2203 face images)
for categories in (list(mask_dir.iterdir())):
    for image_path in categories.iterdir():
        df_object = df_object.append({
            "image": str(image_path),
            "mask": 1
        }, ignore_index=True)
        # print(df_object)
print("masked data indexed")

# label all the images without mask to "0"
# there are 460 subjects without mask. (90000 face images)
for categories in (list(mask_less_dir.iterdir())):
    image_counter = 0
    for image_path in categories.iterdir():
        if image_counter < 5:          # 5 images per folder Max
            df_object = df_object.append({
                "image": str(image_path),
                "mask": 0
            }, ignore_index=True)
            # print(df_object)
            image_counter += 1
        else:
            break
print("without masked data indexed")

# save the dataframe in .csv file
df_to_csv = dataset_dir/"dataset_ver2_short_version.csv"
print("saving Dataframe to: ", df_to_csv)
df_object.to_csv(df_to_csv)
