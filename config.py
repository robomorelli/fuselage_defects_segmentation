import os

root = os.getcwd()
paths_to_exclude = ['data', 'dataset', 'models', 'notebooks', 'notebook',
                    'results', 'preprocessing', 'evaluation', 'adaptation']
paths = []
root_parts = root.split('/')
root = [x if x not in paths_to_exclude else '' for x in root_parts]
root = '/'.join(root)

root_data_path = os.path.join(root,'data')
conf_path = os.path.join(root,'configuration')

# Train data path
data_path = os.path.join(root,'data')
data_images_path = os.path.join(root,'data/images')
data_labels_path = os.path.join(root,'data/labels')

train_data_path = os.path.join(root,'data/train/')

train_images_path = os.path.join(root,'data/train/images')
train_masks_path = os.path.join(root,'data/train/masks')

cropped_train_renamed_masks_path = os.path.join(root,'data/train/cropped_data/renamed_masks')
cropped_train_threshold_results = os.path.join(root,'data/train/cropped_data/threshold_masks')

train_report_path = os.path.join(root,'data/train/report/data_report.csv')
train_filtered_report_path = os.path.join(root,'data/train/filtered/report/data_report.csv')

val_data_path = os.path.join(root,'data/val')
val_images_path = os.path.join(root,'data/val/images')
val_masks_path = os.path.join(root,'data/val/masks')
raw_val_masks_path = os.path.join(root,'data/val/raw_masks')
to_rename_val_masks_path = os.path.join(root,'data/val/segmented_task_masks')

# Cropped train&val
cropped_train_path = os.path.join(root, 'data/train/cropped_data')
cropped_train_masks_path = os.path.join(root,'data/train/cropped_data/masks')
cropped_train_images_path = os.path.join(root, 'data/train/cropped_data/images')

cropped_val_path = os.path.join(root, 'data/val/cropped_data')
cropped_val_images_path = os.path.join(root, 'data/val/cropped_data/images')
cropped_val_masks_path = os.path.join(root, 'data/val/cropped_data/masks')
cropped_val_tot_bkg_images_path = os.path.join(root, 'data/val/cropped_data/tot_bkg_images')
cropped_val_tot_bkg_masks_path = os.path.join(root, 'data/val/cropped_data/tot_bkgmasks')

# Test data path
test_data_path = os.path.join(root,'data/test')

test_images_path = os.path.join(root,'data/test/images')
test_masks_path = os.path.join(root,'data/test/masks')
raw_test_masks_path = os.path.join(root,'data/test/raw_masks')
to_rename_test_masks_path = os.path.join(root,'data/test/segmented_task_masks')


# Cropped test
cropped_test_data_path = os.path.join(root, 'data/test/cropped_data')
cropped_test_renamed_masks_path = os.path.join(root, 'data/test/cropped_data/renamed_masks')

cropped_test_images_path = os.path.join(root, 'data/test/cropped_data/images')
cropped_test_masks_path = os.path.join(root, 'data/test/cropped_data/masks')
model_results = os.path.join(root, 'model_results')

IMG_WIDTH = 4096
IMG_HEIGHT = 3000

class_names = ["Paint Defect", "Drill Start", "Mark", "Drill Run", "Rilavorazione Incorretta",
               "Gouge", "Graffio", "Peeling", "Abrasione", "Cage Mark"]


map_class_label_to_name = {0:"Paint Defect", 1:"Drill Start", 2:"Mark", 3:"Drill Run", 4:"Rilavorazione Incorretta",
                            5:"Gouge", 6:"Graffio", 7:"Peeling", 8:"Abrasione", 9:"Cage Mark"}

