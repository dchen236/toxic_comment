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
    with open(f_name) as read_obj, open(f'cleaned_{f_name}', 'w') as write_obj:
        print(f'Cleaning {f_name}')
        reader = csv.reader(read_obj, delimiter='\t')
        writer = csv.writer(write_obj, delimiter='\t')

        headers = next(reader)
        headers.append('cleaned_comment')
        writer.writerow(headers)
        i = headers.index('body')
        
        for row in reader:
            clean_body = preprocess(row[i])
            row.append(clean_body)
            writer.writerow(row)
            
        
    
