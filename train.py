from torch.utils.data import DataLoader
from datasets import *
import torch.optim as optim
from models import *
from sklearn import model_selection
import config as cfg
import torch.nn as nn
import time
import copy
from metrics import *
from util import save_submission, convert_dataframe_to_bool
# # Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")


def train(model, criterion, opti, dataset_loader, dataset_sizes, device, max_eps):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    model = model.to(device=device)
    for ep in range(max_eps):
        print("Epoch{}/{}".format(ep + 1, max_eps))
        print("-" * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                #                 scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0
            running_corrects = 0

            for (comments, attn_masks, labels) in dataset_loader[phase]:
                opti.zero_grad()
                comments = comments.to(device=device)
                attn_masks = attn_masks.to(device=device)
                labels = labels.to(device=device)
                with torch.set_grad_enabled(phase == 'train'):
                    # forward
                    outputs = model(comments, attn_masks)
                    preds = torch.sigmoid(outputs.unsqueeze(-1)).long()
                    loss = criterion(outputs.squeeze(-1), labels.float())
                if phase == 'train':
                    loss.backward()
                    opti.step()

                running_loss += loss.item() * comments.size(0)
                preds_toxic = preds.squeeze() > 0.5
                labels_toxic = labels > 0.5
                running_corrects += torch.sum(preds_toxic == labels_toxic)

            epoch_loss = running_loss / dataset_sizes[phase]
            accuracy = running_corrects.double() / dataset_sizes[phase]

            print("%s Loss is: %f" % (phase, epoch_loss))
            print("Accuracy is %f" % accuracy)

            if phase == 'val' and accuracy > best_acc:
                print("updating best model")
                best_acc = accuracy
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

        time_elapsed = time.time() - since
        print("Trainning time: %d min %f sec" % (time_elapsed // 60, time_elapsed % 60))
        print("Best accuracy is %f" % best_acc)
        model.load_state_dict(best_model_wts)
        return model, best_acc


def bert_predict(model, validation_loader, device):
    predictions = []
    for (comments, attn_masks, labels) in validation_loader:
        comments = comments.to(device=device)
        attn_masks = attn_masks.to(device=device)
        with torch.set_grad_enabled(False):
            # forward
            outputs = model(comments, attn_masks)
            preds = torch.sigmoid(outputs.unsqueeze(-1)).long()
            predictions.extend(preds.reshape(-1).cpu().detach().numpy())
    return predictions


def bert_train():
    TRAIN_CSV = cfg.train_csv
    TEST_CSV = cfg.test_csv

    TOXICITY_COLUMN = cfg.toxicity_column
    TEXT_COLUMN = cfg.text_column
    IDENTITY_COLUMN = cfg.identity_columns
    BINARY_TOXICITY = "binary_toxicity"
    SUBMISSION_CSV = "bert_sumbission.csv"
    MODEL_NAME = 'Bert_model'

    train_df = pd.read_csv(TRAIN_CSV)
    train_df = convert_dataframe_to_bool(train_df, TOXICITY_COLUMN, IDENTITY_COLUMN)
    train_df[TEXT_COLUMN] = train_df[TEXT_COLUMN].astype(str)
    train_df[BINARY_TOXICITY] = train_df[TOXICITY_COLUMN] >= 0.5

    test_df = pd.read_csv(TRAIN_CSV)
    test_df[TEXT_COLUMN] = train_df[TEXT_COLUMN].astype(str)

    print('loaded %d train df' % len(train_df))
    # Make sure all comment_text values are strings

    train_df, validate_df = model_selection.train_test_split(train_df,
                                                             test_size=0.2,
                                                             random_state=123,
                                                             stratify=train_df[BINARY_TOXICITY])

    train_set = toxic_Dataset_Bert(input=train_df)
    val_set = toxic_Dataset_Bert(input=validate_df)
    test_set = toxic_Dataset_Bert(input=test_df)

    train_loader = DataLoader(train_set, batch_size=128, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=128, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=128, num_workers=8)

    dataset_loader = {'train': train_loader,
                      'val': val_loader}

    dataset_sizes = {'train': train_df.shape[0],
                     'val': validate_df.shape[0]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Toxic_Classifier_Bert(freeze_bert=True)
    criterion = nn.MSELoss()
    opti = optim.Adam(model.parameters(), lr=2e-5)
    
    bert_model, acc = train(model, criterion, opti, dataset_loader, dataset_sizes, device, max_eps=1)
    validate_df[MODEL_NAME] = bert_predict(bert_model, val_loader, device)
    bias_metrics_df = compute_bias_metrics_for_model(validate_df, IDENTITY_COLUMN, MODEL_NAME, TOXICITY_COLUMN)
    get_final_metric(bias_metrics_df, calculate_overall_auc(validate_df, MODEL_NAME))
    testset_predictions = bert_predict(bert_model, test_loader, device)
    save_submission(testset_predictions, SUBMISSION_CSV)


if __name__ == "__main__":
    bert_train()