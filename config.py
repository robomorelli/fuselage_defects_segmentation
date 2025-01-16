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
masks_viz_path = os.path.join(root,'data/visualization')
full_size_masks_classes_path = os.path.join(root,'data/masks_classes')
cropped_data_path = os.path.join(root,'data/cropped_data')
cropped_images_path = os.path.join(root,'data/cropped_data/images')
cropped_tot_bkg_data_path = os.path.join(root,'data/cropped_data', 'tot_bkg')

fine_tuning_data_path = os.path.join(root,'fine_tuning_data')
fine_tuning_train_images_path = os.path.join(root,'fine_tuning_data/train/images')
fine_tuning_train_masks_path = os.path.join(root,'fine_tuning_data/train/masks')
fine_tuning_val_images_path = os.path.join(root,'fine_tuning_data/val/images')
fine_tuning_val_masks_path = os.path.join(root,'fine_tuning_data/val/masks')

fine_tuning_train_cropped_images_path = os.path.join(root,'fine_tuning_data/train/cropped_images')
fine_tuning_train_cropped_masks_path = os.path.join(root,'fine_tuning_data/train/cropped_masks')
fine_tuning_val_cropped_images_path = os.path.join(root,'fine_tuning_data/val/cropped_images')
fine_tuning_val_cropped_masks_path = os.path.join(root,'fine_tuning_data/val/cropped_masks')

cropped_tot_bkg_images_path = os.path.join(root,'data/cropped_data', 'tot_bkg/images')

data_images_path = os.path.join(root,'data/images')
data_labels_path = os.path.join(root,'data/labels')
data_masks_path = os.path.join(root,'data/masks')
data_masks_npy_path = os.path.join(root,'data/masks_npy')

test_1_data_path = os.path.join(root,'data/test_1/')
test_1_data_images_path = os.path.join(root,'data/test_1/images')
test_1_data_masks_path = os.path.join(root,'data/test_1/masks')
test_1_data_renamed_masks_path = os.path.join(root,'data/test_1/renamed_masks')
cropped_test_1_data_path = os.path.join(root,'data/test_1/cropped_data')
cropped_test_1_images_path = os.path.join(root,'data/test_1/cropped_data/images')
cropped_test_1_masks_path = os.path.join(root,'data/test_1/cropped_data/masks')

april_data_path = os.path.join(root,'data/april/')
april_data_images_path = os.path.join(root,'data/april/images')
april_data_masks_path = os.path.join(root,'data/april/masks')
april_data_renamed_masks_path = os.path.join(root,'data/april/renamed_masks')
cropped_april_data_path = os.path.join(root,'data/april/cropped_data')
cropped_april_images_path = os.path.join(root,'data/april/cropped_data/images')
cropped_april_masks_path = os.path.join(root,'data/april/cropped_data/masks')
april_model_results = os.path.join(root,'model_results/segformer_k_fold_multiclass/nvidia/mit-b5/fold_4/segformer_processor_decoder_w_1_3_2_2024_03_27_14_31_29/april/merged_model_results')

may_data_path = os.path.join(root,'data/may/')
may_data_images_path = os.path.join(root,'data/may/images')
may_data_masks_path = os.path.join(root,'data/may/masks')
may_data_renamed_masks_path = os.path.join(root,'data/may/renamed_masks')
cropped_may_data_path = os.path.join(root,'data/may/cropped_data')
cropped_may_images_path = os.path.join(root,'data/may/cropped_data/images')
cropped_may_masks_path = os.path.join(root,'data/may/cropped_data/masks')
may_model_results = os.path.join(root,'model_results/segformer_k_fold_multiclass/nvidia/mit-b5/fold_4/segformer_processor_decoder_w_1_3_2_2024_03_27_14_31_29/may/merged_model_results')

test_original_data_path = os.path.join(root,'data/original_test/')
test_original_images_path = os.path.join(root,'data/original_test/images')
test_original_masks_path = os.path.join(root,'data/original_test/masks')
test_original_renamed_masks_path = os.path.join(root,'data/original_test/renamed_masks')
cropped_test_original_data_path = os.path.join(root,'data/original_test/cropped_data')
cropped_test_original_images_path = os.path.join(root,'data/original_test/cropped_data/images')
cropped_test_original_masks_path = os.path.join(root,'data/original_test/cropped_data/masks')

feb_mar_apr_may_model_results = os.path.join(root,'model_results/segformer_k_fold_multiclass/nvidia/mit-b5/fold_4/segformer_processor_decoder_w_1_3_2_2024_03_27_14_31_29/feb_mar_apr_may/merged_model_results')


multiclass_masks_path = os.path.join(root,'data/masks_classes')
data_masks_comparison_path = os.path.join(root,'data/masks_comparison')
renamed_masks_path = os.path.join(root,'data/renamed_masks')

to_rename_masks_path = os.path.join(root,'data/segmented_task_masks')

train_data_path = os.path.join(root,'data/train/')

train_images_path = os.path.join(root,'data/train/images')
train_masks_path = os.path.join(root,'data/train/masks')
common_path_train_results = os.path.join(root,'results/train')
common_path_val_results = os.path.join(root,'results/val')
common_path_test_results = os.path.join(root,'results/test')
common_path_train_results_viz = os.path.join(root,'results_viz/train')
common_path_val_results_viz = os.path.join(root,'results_viz/val')
common_path_test_results_viz = os.path.join(root,'results_viz/test')

k_fold_data_path = os.path.join(root,'data/k-fold_archive')
split1_k_fold_data_path = os.path.join(root,'data/k-fold_archive/fold_1')

cropped_train_renamed_masks_path = os.path.join(root,'data/train/cropped_data/renamed_masks')
cropped_train_threshold_results = os.path.join(root,'data/train/cropped_data/threshold_masks')

cropped_tot_bkg_renamed_masks_path = os.path.join(root, 'data/cropped_data/tot_bkg/renamed_masks')
cropped_renamed_masks_path = os.path.join(root, 'data/cropped_data/renamed_masks')

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
cropped_val_tot_bkg_images_path = os.path.join(root, 'data/val/cropped_data/tot_bkg/images')
cropped_val_tot_bkg_masks_path = os.path.join(root, 'data/val/cropped_data/tot_bkg/masks')


# Test data path
test_data_path = os.path.join(root,'data/test')

test_images_path = os.path.join(root,'data/test/images')
test_masks_path = os.path.join(root,'data/test/masks')
raw_test_masks_path = os.path.join(root,'data/test/raw_masks')
to_rename_test_masks_path = os.path.join(root,'data/test/segmented_task_masks')

# Cropped test1
cropped_test_data_path = os.path.join(root, 'data/test/cropped_data')
cropped_test_renamed_masks_path = os.path.join(root, 'data/test/cropped_data/renamed_masks')

cropped_test_images_path = os.path.join(root, 'data/test/cropped_data/images')
cropped_test_masks_path = os.path.join(root, 'data/test/cropped_data/masks')
model_results = os.path.join(root, 'model_results')

masks_to_remap_path = os.path.join(root, 'data/masks_to_remap')
images_2_wave_path = os.path.join(root, 'data/images_2_wave')

IMG_WIDTH = 4096
IMG_HEIGHT = 3000

class_names = ["Paint Defect", "Drill Start", "Mark", "Drill Run", "Rilavorazione Incorretta",
               "Gouge", "Graffio", "Peeling", "Abrasione", "Cage Mark"]


map_class_label_to_name = {0:"Paint Defect", 1:"Drill Start", 2:"Mark", 3:"Drill Run", 4:"Rilavorazione Incorretta",
                            5:"Gouge", 6:"Graffio", 7:"Peeling", 8:"Abrasione", 9:"Cage Mark"}


custom_list = ["0000_label_iSVx22hl.png", "0000_label_KhubE3m1.png", "0000_label_lY40XnMe.png"
               ,"0000_label_Pe712dpb.png", "0000_label_uCXpTeJJ.png", "0000_label_UdQfh0C1.png"
               , "0000_label_XP0UKXUr.png", "0001_label_wmLYCo2j.png", "0002_label_ouHctxiB.png"
               ,"0006_label_KhubE3m1.png", "0007_label_5caRcr55.png", "00000008_1.png", "0008_label_KhubE3m1.png"]

