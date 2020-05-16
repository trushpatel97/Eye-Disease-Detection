Retina-AI

Dataset Collection:
	Collect a dataset and split into directories for matching categories
	with an all uppercase name inside each of 3 parent directories labeled
	'train', 'test', and 'val'.

For Example:
	When downloading the included OCT dataset, you will notice it contains
	2 folders (train, test). First, you must move some images from the training or testing sets into a new folder called 'val'. Inside each are 4 folders with their
	corresponding category (CNV, DME, DRUSEN, NORMAL). The image path expected
	at command-line matches that of the folder containing the first 3 folders
	(train, test, val).

Sample Usage:
	python retrain.py --images /path/to/images 

Generating ROC:
  Uncomment the last few lines in the main function of the retrain.py file. Change [LIST_OF_POS_IDX] with
  a list of indices of the positive categories (per the output_labels.txt file). Run the script.

Occlusion:
        python occlusion.py
                --image_dir /path/to/image
                --graph /tmp/output_graph.pb
                --labels /tmp/output_labels.txt

