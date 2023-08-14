import os
import torch

def load_train(data_path, device):

    train_data_path = os.path.join(data_path, "train.pt")
    train_label_path = os.path.join(data_path, "train_labels.pt")
    val_data_path = os.path.join(data_path, "val.pt")
    val_label_path = os.path.join(data_path, "val_labels.pt")

    #print(train_data_path)

    train_data = torch.load(train_data_path, map_location=device)
    train_label = torch.load(train_label_path, map_location=device)
    val_data = torch.load(val_data_path, map_location=device)
    val_label = torch.load(val_label_path, map_location=device)

    return train_data, train_label, val_data, val_label

def load_test(data_path, running_condition, device):
    if running_condition == 1:
        test_1_1 = torch.load(os.path.join(data_path, "test_1_1.pt"), map_location=device)
        label_1_1 = torch.load(os.path.join(data_path, "label_1_1.pt"), map_location=device)

        test_1_2 = torch.load(os.path.join(data_path, "test_1_2.pt"), map_location=device)
        label_1_2 = torch.load(os.path.join(data_path, "label_1_2.pt"), map_location=device)

        test_1_3 = torch.load(os.path.join(data_path, "test_1_3.pt"), map_location=device)
        label_1_3 = torch.load(os.path.join(data_path, "label_1_3.pt"), map_location=device)

        test_1_4 = torch.load(os.path.join(data_path, "test_1_4.pt"), map_location=device)
        label_1_4 = torch.load(os.path.join(data_path, "label_1_4.pt"), map_location=device)

        test_1_5 = torch.load(os.path.join(data_path, "test_1_5.pt"), map_location=device)
        label_1_5 = torch.load(os.path.join(data_path, "label_1_5.pt"), map_location=device)

        test_1_6 = torch.load(os.path.join(data_path, "test_1_6.pt"), map_location=device)
        label_1_6 = torch.load(os.path.join(data_path, "label_1_6.pt"), map_location=device)

        test_1_7 = torch.load(os.path.join(data_path, "test_1_7.pt"), map_location=device)
        label_1_7 = torch.load(os.path.join(data_path, "label_1_7.pt"), map_location=device)

        return (test_1_1, label_1_1), (test_1_2, label_1_2), (test_1_3, label_1_3), (test_1_4, label_1_4), (test_1_5, label_1_5), (test_1_6, label_1_6), (test_1_7, label_1_7)
    
    elif running_condition == 2:
        test_2_1 = torch.load(os.path.join(data_path, "test_2_1.pt"), map_location=device)
        label_2_1 = torch.load(os.path.join(data_path, "label_2_1.pt"), map_location=device)

        test_2_2 = torch.load(os.path.join(data_path, "test_2_2.pt"), map_location=device)
        label_2_2 = torch.load(os.path.join(data_path, "label_2_2.pt"), map_location=device)

        test_2_3 = torch.load(os.path.join(data_path, "test_2_3.pt"), map_location=device)
        label_2_3 = torch.load(os.path.join(data_path, "label_2_3.pt"), map_location=device)

        test_2_4 = torch.load(os.path.join(data_path, "test_2_4.pt"), map_location=device)
        label_2_4 = torch.load(os.path.join(data_path, "label_2_4.pt"), map_location=device)

        test_2_5 = torch.load(os.path.join(data_path, "test_2_5.pt"), map_location=device)
        label_2_5 = torch.load(os.path.join(data_path, "label_2_5.pt"), map_location=device)

        test_2_6 = torch.load(os.path.join(data_path, "test_2_6.pt"), map_location=device)
        label_2_6 = torch.load(os.path.join(data_path, "label_2_6.pt"), map_location=device)

        test_2_7 = torch.load(os.path.join(data_path, "test_2_7.pt"), map_location=device)
        label_2_7 = torch.load(os.path.join(data_path, "label_2_7.pt"), map_location=device)

        return (test_2_1, label_2_1), (test_2_2, label_2_2), (test_2_3, label_2_3), (test_2_4, label_2_4), (test_2_5, label_2_5), (test_2_6, label_2_6), (test_2_7, label_2_7)
    
    else:
        test_3_1 = torch.load(os.path.join(data_path, "test_3_1.pt"), map_location=device)
        label_3_1 = torch.load(os.path.join(data_path, "label_3_1.pt"), map_location=device)

        test_3_2 = torch.load(os.path.join(data_path, "test_3_2.pt"), map_location=device)
        label_3_2 = torch.load(os.path.join(data_path, "label_3_2.pt"), map_location=device)

        test_3_3 = torch.load(os.path.join(data_path, "test_3_3.pt"), map_location=device)
        label_3_3 = torch.load(os.path.join(data_path, "label_3_3.pt"), map_location=device)

        return (test_3_1, label_3_1), (test_3_2, label_3_2), (test_3_3, label_3_3)