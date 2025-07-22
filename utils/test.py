#%%
import jieba
import jieba.analyse
from optparse import OptionParser
from sheet import Sheet
import regex as re

def preprocess_text(text):
    # remove all var tags
    text = re.sub(r'\[var(\d)?\]', '', text)
    # remove all xml tags
    text = re.sub(r'<[^>]+>', '', text)
    return text

def preprocess_tag(tag):
    # remove any numbers
    match = re.search(r'([\+\-]?\d(\.\d+)?\%?)+', tag)
    if match:
        tag = tag.replace(match.group(), '')
    
    return tag

count = 0
startidx = 429
length = 100
sheet = Sheet('TLON语言包ES 20250701.xlsx')
jieba.load_userdict('terms_dict.txt')

for i in range(startidx, startidx+length):
    (cn, es) = sheet[i]
    if not isinstance(cn, str):
        continue

    cn_new = preprocess_text(cn)

    tags = jieba.analyse.extract_tags(cn_new, topK=100)
    tags = [preprocess_tag(t) for t in tags]
    tags = [t for t in tags if t is not None and t != '']

    if len(tags) == 0:
        continue
    
    print((cn, cn_new, es))
    if len(tags) != 0:
        print(tags)

    print('-' * 50)
    count += 1
    if count > 100:
        break

#%%
s = "nt"
s.startswith