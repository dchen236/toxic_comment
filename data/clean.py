import csv
import re


def preprocess(data):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    data = clean_special_chars(str(data), punct)
    return data

f_names = ['askreddit.csv', 'changemyview.csv',  'politics.csv',  'wholesomememes.csv']


for f_name in f_names:
    with open(f_name) as read_obj, open(f'predict_{f_name}', 'w') as write_obj:
        print(f'Cleaning {f_name}')
        reader = csv.reader(read_obj, delimiter='\t')
        writer = csv.writer(write_obj)

        headers = next(reader)
        h = headers.index('body')
        i = headers.index('id')
        writer.writerow(['id', 'comment'])        
        for row in reader:
            id = row[i]
            body = row[h]
            clean_body = preprocess(body)
            output_row = [id, clean_body]
            writer.writerow(output_row)
            
        
    
