import os


root_path = os.path.dirname(os.path.dirname(__file__))
share_path = os.path.join(root_path, 'share')
data_path = os.path.join(root_path, 'data')

digit5k_dataset_path = os.path.join(data_path, 'digits.png')

digit_dataset_path = os.path.join(data_path, 'digit_dataset')
extracted_digits_path = os.path.join(data_path, 'extracted_digits')
digit_samples_path = os.path.join(data_path, 'digit_samples')

sudokus_path = os.path.join(data_path, 'sudokus')
sudokus_gt_path = os.path.join(sudokus_path, 'ground_truth_new.csv')

classifier_checkpoint_path = os.path.join(share_path, 'digit_classifier.pth')
