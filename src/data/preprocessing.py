import os
import json
import pandas as pd

def load_annotations_df(dataset_root, split_name, json_filename='_annotations.coco.json'):
    """Loads COCO JSON and returns annotations DataFrame and category dict."""
    json_path = os.path.join(dataset_root, split_name, json_filename)
    if not os.path.exists(json_path):
        print(f"Error: Annotation file not found at {json_path}")
        return pd.DataFrame(), {}, {}
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    annotations_list = coco_data['annotations']
    categories_dict = {cat['id']: cat['name'] for cat in coco_data['categories']}
    return pd.DataFrame(annotations_list), categories_dict, coco_data

def remove_civilian_cars(json_path, output_path, class_to_remove=2):
    """Removes a specific class (e.g. Civilian Cars) from the annotations."""
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # 1. Filter categories
    new_categories = [cat for cat in coco_data['categories'] if cat['id'] != class_to_remove]

    # 2. Filter annotations
    new_annotations = [ann for ann in coco_data['annotations'] if ann['category_id'] != class_to_remove]

    keep_image_ids = set(ann['image_id'] for ann in new_annotations)

    # 3. Filter images
    new_images = [img for img in coco_data['images'] if img['id'] in keep_image_ids]

    final_annotations = [ann for ann in new_annotations if ann['image_id'] in keep_image_ids]

    new_coco = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": new_categories,
        "images": new_images,
        "annotations": final_annotations
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(new_coco, f, indent=4)
        
    return new_images, final_annotations

def update_coco_json(json_path, new_category_mapping, new_categories_list, output_filename='remapped_annotations.json'):
    """Updates category_ids to be contiguous."""
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    updated_annotations = []
    for ann in coco_data['annotations']:
        old_cat_id = ann['category_id']
        if old_cat_id in new_category_mapping:
            ann['category_id'] = new_category_mapping[old_cat_id]
            updated_annotations.append(ann)

    coco_data['categories'] = new_categories_list
    coco_data['annotations'] = updated_annotations

    output_dir = os.path.dirname(json_path)
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=4)
    return output_path

def clean_yolo_labels(dataset_dir, folder_name, class_to_remove=1):
    """Cleans YOLO .txt labels for a specific class (Used in YOLO preprocessing)."""
    labels_path = os.path.join(dataset_dir, folder_name, 'labels')
    if not os.path.exists(labels_path):
        return

    for filename in os.listdir(labels_path):
        if not filename.endswith('.txt'):
            continue

        file_path = os.path.join(labels_path, filename)
        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            class_id = int(parts[0])

            if class_id == class_to_remove:
                continue

            if class_id > class_to_remove:
                class_id -= 1

            parts[0] = str(class_id)
            new_lines.append(" ".join(parts) + "\n")

        with open(file_path, 'w') as f:
            f.writelines(new_lines)
