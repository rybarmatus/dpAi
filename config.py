img_h = 100
img_w = 100
img_h_fine = 150
img_w_fine = 150

batch_size = 32

# cesta k obrazkom stranok
images_path = "D:\\dp2\\web_categories - Copy"

# extrahovane priznaky s rozmerom stranok 100x100
featuresFile = 'features_reduced.csv'

# extrahovane priznaky s rozmerom stranok 100x100 scalenute na 0-1
features_files_std = 'features_reduced_std.csv'

# subor s HTML subormi stranok
html_folder = 'D:\\storedHtmls\\web_categories - Copy'

# csv s textom z web stranok

web_texts = 'web_texts.csv'

purpose_image_path = 'E:\\functional'
purpose_weights_name = 'fine_tuned_categorical_purpose.h5'

binary_image_path = 'D:\\detail_list_data'
binary_weights_name = 'fine_tuned_binary.h5'

all_categories_image_path = 'D:\\all categories\\web_categories - Copy'
all_categories_weight_name = 'fine_tuned_all_cat.h5'