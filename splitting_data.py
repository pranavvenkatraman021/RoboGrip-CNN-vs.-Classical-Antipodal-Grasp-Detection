#imports
import os
import glob
import random
import json


#set fixed random seed
random.seed(42)


#scan subfolders and get every unique object ID
def get_ids():
   pcd_ids = []
   root = "/Users/pranavvenkatraman/Downloads/Cornell Grasp Data"
   #finds all subfolders inside root
   subfolders = sorted(glob.glob(os.path.join(root, "*")))


   for folder in subfolders:
       if not os.path.isdir(folder):
           continue
       #each rgb image file
       rgb_files = glob.glob(os.path.join(folder, "*r.png"))


       #isolate ID number
       for rgb_path in rgb_files:
           filename = os.path.basename(rgb_path)
           pcd_id = filename.replace("pcd", "").replace("r.png", "")
           pcd_ids.append({"id": pcd_id, "folder": folder})
  
   return pcd_ids


#shuffle and split into train, validation, test
def split_data(pcd_ids, train = 0.70, val = 0.15):


   shuffled = pcd_ids.copy()
   random.shuffle(shuffled)


   #cutoff points
   total = len(shuffled)
   train_val = int(total * train)
   val_val = train_val + int(total * val)


   #split
   train_set = shuffled[: train_val]
   val_set = shuffled[train_val : val_val]
   test_set = shuffled[val_val :]


   return train_set, val_set, test_set


if __name__ == "__main__":
   id_list = get_ids()
   train_set, val_set, test_set = split_data(id_list)


   print(f"total: {len(id_list)}")
   print(f"train: {len(train_set)}")
   print(f"validation: {len(val_set)}")
   print(f"test: {len(test_set)}")


   #save to JSON file
   split_output = {
       "train": train_set,
       "val": val_set,
       "test": test_set
   }


   with open("dataset_split.json", "w") as f:
       json.dump(split_output, f, indent=2)


   print("Saved to json")

