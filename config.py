covert_glove_to_dict = False # set this to true if glove embedding is not converted to dictionary
verbose = True
glove = {
        "embedding_path": 'glove-embeddings/glove.6B.100d.txt',
        "save_dict_path": "glove-embeddings/glove.6B.100d_dict.pkl",
        "dimension":100
        }


toxicity_column = 'target'
text_column = 'comment_text'
train_csv = 'data/train.csv'
test_csv = 'data/test.csv'
sample_submission_csv = "data/sample_submission.csv"

# List all identities
identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

max_seq_len = 250 # make sure the sentences have the same length


