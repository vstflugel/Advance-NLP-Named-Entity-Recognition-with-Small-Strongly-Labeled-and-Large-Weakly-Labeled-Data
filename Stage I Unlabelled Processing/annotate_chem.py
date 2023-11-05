# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T04:29:38.929503Z","iopub.execute_input":"2023-10-31T04:29:38.929922Z","iopub.status.idle":"2023-10-31T04:29:39.059761Z","shell.execute_reply.started":"2023-10-31T04:29:38.929887Z","shell.execute_reply":"2023-10-31T04:29:39.057926Z"}}
import numpy as np
import random
from itertools import chain
from tqdm import tqdm
import ray


ray.init(ignore_reinit_error=True)

random.seed(0)
np.random.seed(seed=0)

sampling_rate_no_weak_labels = 0.1

num = 0
all_text = []
with open('all_text.txt', 'r') as f:
    for l in f:
        l = l.strip()
        if l == "":
            continue
        all_text.append(l)
        num += 1

# Change the TGT_ENTITY_TYPE for generating weakly supervised data for different entity
# TGT_ENTITY_TYPE = 'Disease'
TGT_ENTITY_TYPE = 'Chemical'

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T04:29:40.106483Z","iopub.execute_input":"2023-10-31T04:29:40.106920Z","iopub.status.idle":"2023-10-31T04:29:40.114725Z","shell.execute_reply.started":"2023-10-31T04:29:40.106887Z","shell.execute_reply":"2023-10-31T04:29:40.113443Z"}}
all_text[0]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T04:29:40.910193Z","iopub.execute_input":"2023-10-31T04:29:40.910603Z","iopub.status.idle":"2023-10-31T04:29:40.923760Z","shell.execute_reply.started":"2023-10-31T04:29:40.910573Z","shell.execute_reply":"2023-10-31T04:29:40.922429Z"}}
with open('disease_dict.txt', 'r') as f:
    dict_disease = [x.strip() for x in f if x.strip() != ""]
with open('chem_dict.txt', 'r') as f:
    dict_chem = [x.strip() for x in f if x.strip() != ""]
print(len(dict_chem))
print(len(dict_disease))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T04:29:41.638927Z","iopub.execute_input":"2023-10-31T04:29:41.639362Z","iopub.status.idle":"2023-10-31T04:30:54.812472Z","shell.execute_reply.started":"2023-10-31T04:29:41.639315Z","shell.execute_reply":"2023-10-31T04:30:54.811190Z"}}
@ray.remote
def f(text_lines):
    labeled_lines = []
    unlabeled_lines = []

    for line in text_lines:
        labels = ["O"]*len(line)
        if TGT_ENTITY_TYPE == "Chemical":
            entities = dict_chem
        if TGT_ENTITY_TYPE == "Disease":
            entities = dict_disease
        entity_type = TGT_ENTITY_TYPE

        for entity in entities:
            en_len = len(entity)
            p = line.find(entity)
            while p != -1:
                if (p>0 and line[p-1].isalnum()) or (p+en_len<len(line) and line[p+en_len].isalnum()):
                    # only part of string skip
                    pass
                elif all([l=='O' for l in labels[p:p+en_len]]):
                    for i in range(1, en_len):
                        labels[p+i] = 'I-'+entity_type
                    labels[p] = 'B-'+entity_type
                    if p>0 and line[p-1] == "-":
                        pp = p-1
                        while pp >= 0 and (line[pp].isalnum() or line[pp] == '-'):
                            assert labels[pp]=='O', f"AS1\n{line[p:p+en_len]}\n{labels[p:p+en_len]}\n{line[pp:p+en_len]}\n{labels[pp:p+en_len]}\n{line[:p+en_len]}\n{labels[:p+en_len]}"
                            labels[pp+1] = 'I-'+entity_type
                            labels[pp] = 'B-'+entity_type
                            pp -= 1
                    if p+en_len<len(line) and line[p+en_len] == "-":
                        pp = p+en_len
                        while pp<len(line) and (line[pp].isalnum() or line[pp] == '-'):
#                             assert labels[pp]=='O', f"AS2\n{line[p:p+en_len]}\n{labels[p:p+en_len]}\n{line[p:pp]}\n{labels[p:pp]}\n{line[:pp]}\n{labels[:pp]}"
                            labels[pp] = 'I-'+entity_type
                            pp += 1
                            
                elif all([l[0]=='I' for l in labels[p:p+en_len]]):
                    # contained
                    pass
                elif labels[p][0] == 'B' and all([l[0]=='I' for l in labels[p+1:p+en_len]]):
                    # contained
                    pass
                elif all([l in ['O', 'I-'+entity_type] for l in labels[p:p+en_len]]):
                    # partially contained, extend current entity
                    for i in range(en_len):
                        labels[p+i] = 'I-'+entity_type
                    if p+en_len<len(line) and line[p+en_len] == "-":
                        pp = p+en_len
                        while pp<len(line) and (line[pp].isalnum() or line[pp] == '-'):
#                             assert labels[pp]=='O', f"AS3\n{line[p:p+en_len]}\n{labels[p:p+en_len]}\n{line[p:pp]}\n{labels[p:pp]}\n{line[:pp]}\n{labels[:pp]}"
                            labels[pp] = 'I-'+entity_type
                            pp += 1
                elif all([l in ['O', 'I-'+entity_type, 'B-'+entity_type] for l in labels[p:p+en_len]]):
                    # partially contained, extend current entity
                    for i in range(1,en_len):
                        labels[p+i] = 'I-'+entity_type
                        
                    if p+en_len<len(line) and line[p+en_len] == "-" and labels[p+en_len]=='O':
                        pp = p+en_len
                        while pp<len(line) and (line[pp].isalnum() or line[pp] == '-'):
#                             assert labels[pp]=='O', f"AS4\n{line[p:p+en_len]}\n{labels[p:p+en_len]}\n{line[p:pp]}\n{labels[p:pp]}\n{line[:pp]}\n{labels[:pp]}"
                            labels[pp] = 'I-'+entity_type
                            pp += 1
                            
                    if labels[p] != 'I-'+entity_type:
                        labels[p] = 'B-'+entity_type
                        if p>0 and line[p-1] == "-":
                            pp = p-1
                            while pp >= 0 and (line[pp].isalnum() or line[pp] == '-'):
                                assert labels[pp]=='O', f"AS5\n{line[p:p+en_len]}\n{labels[p:p+en_len]}\n{line[pp:p+en_len]}\n{labels[pp:p+en_len]}\n{line[:p+en_len]}\n{labels[:p+en_len]}"
                                labels[pp+1] = 'I-'+entity_type
                                labels[pp] = 'B-'+entity_type
                                pp -= 1
                else:
                    assert False, f"AS6\nsomething wrong\n{labels[p:p+en_len]} \n{line[p:p+en_len]}\t{entity_type}\n{line}\n{labels}"
                p = line.find(entity, p+1)
        if all([l=='O' for l in labels]):
            unlabeled_lines.append([line, labels])
        else:
            labeled_lines.append([line, labels])
    
    return labeled_lines, unlabeled_lines

# labeled_lines,unlabeled_lines = f(all_text)
    
num_chunks = 500
chunk_size = len(all_text)//num_chunks + 1
all_processed_data = ray.get([f.remote(all_text[chunk_size*i: min(len(all_text), chunk_size*(i+1))]) for i in range(num_chunks)])
labeled_lines = list(chain.from_iterable([x[0] for x in all_processed_data]))
unlabeled_lines = list(chain.from_iterable([x[1] for x in all_processed_data]))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T04:30:54.820019Z","iopub.execute_input":"2023-10-31T04:30:54.820577Z","iopub.status.idle":"2023-10-31T04:30:54.827368Z","shell.execute_reply.started":"2023-10-31T04:30:54.820523Z","shell.execute_reply":"2023-10-31T04:30:54.826103Z"}}
print(len(labeled_lines))
print(len(unlabeled_lines))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T04:30:54.828996Z","iopub.execute_input":"2023-10-31T04:30:54.829478Z","iopub.status.idle":"2023-10-31T04:30:55.457691Z","shell.execute_reply.started":"2023-10-31T04:30:54.829425Z","shell.execute_reply":"2023-10-31T04:30:55.456639Z"}}
import pickle

with open('labeled_lines.pickle', 'wb') as handle:
    pickle.dump(labeled_lines, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('unlabeled_lines.pickle', 'wb') as handle:
    pickle.dump(unlabeled_lines, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T04:30:55.460923Z","iopub.execute_input":"2023-10-31T04:30:55.461433Z","iopub.status.idle":"2023-10-31T04:31:01.912598Z","shell.execute_reply.started":"2023-10-31T04:30:55.461388Z","shell.execute_reply":"2023-10-31T04:31:01.910720Z"}}
## Tokenize

import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
from nltk.corpus import wordnet
from collections import defaultdict
from collections import OrderedDict
from tqdm import tqdm

import re
TOKENIZATION_REGEXS = OrderedDict([
    # NERsuite-like tokenization: alnum sequences preserved as single
    # tokens, rest are single-character tokens.
    ('default', re.compile(r'([^\W_]+|.)')),
    # Finer-grained tokenization: also split alphabetical from numeric.
    ('fine', re.compile(r'([0-9]+|[^\W0-9_]+|.)')),
    # Whitespace tokenization
    ('space', re.compile(r'(\S+)')),
])


def sentence_to_tokens(text, tokenization_re=None):
    """Return list of tokens in given sentence using NERsuite tokenization."""

    if tokenization_re is None:
        tokenization_re = TOKENIZATION_REGEXS.get('default')
    tok = [t for t in tokenization_re.split(text) if t]
    assert ''.join(tok) == text
    return tok

def span_anno2single_anno(token, anno):
    is_O = all([a=='O' for a in anno])
    if is_O:
        return 'O'
    is_B = all([a[0]=='I' for a in anno[1:]]) and anno[0][0] == 'B'
    if is_B:
        return anno[0]
    is_I = all([a[0]=='I' for a in anno])
    if is_I:
        return anno[0]
    print(f"{anno}\n{token}")
    
    pos_l = set()
    for tmp in wordnet.synsets(token):
        if tmp.name().split('.')[0] == token:
            pos_l.add(tmp.pos())
    pos_l = list(pos_l)
#     print(pos_l[0])
    if len(pos_l) == 1 and pos_l[0] in ['a', 's']:
        print(f"===> O (pos: {pos_l[0]})")
        return "O"
    if len(pos_l) != 0:
        print(f"  (pos: {pos_l})")
        
    if any([a[0]=='B' for a in anno]):
        for a in anno:
            if a[0] == 'B':
                print("===> ", a)
                return a
    elif anno[0][0] == 'I':
        print("===> ", anno[0])
        return anno[0]
    else:
        assert False
        

# @ray.remote
def f(text_lines):
    all_samples = []
    for text,anno in tqdm(text_lines):
        # doc2sentence
        sentence_spans = [list(s) for s in sent_detector.span_tokenize(text)]
        new_sentence_spans = []
        for span_id, span in enumerate(sentence_spans):
            if span[1] < len(text)-1:
                if anno[span[1]] != "O":
                    print(f"\'{text[span[1]+1]}\', anno: {anno[span[1]+1]} \n {text[span[0]:span[1]]} \n {span} \n {sentence_spans}")
                    span[1] = sentence_spans[span_id+1][1]
                    sentence_spans[span_id+1][0] = span[0]
                    continue
            new_sentence_spans.append(span)

        sentence_spans = new_sentence_spans
        text_anno_pairs = [(text[span[0]:span[1]], anno[span[0]:span[1]]) for span in sentence_spans ]
        for sent, s_anno in text_anno_pairs:
            tokens = sentence_to_tokens(sent)
            offset = 0
            token_anno_pairs = []
            prev_t_anno = 'O'

            if all([a=="O" for a in s_anno]):
                continue

            for t in tokens:
                if not t.isspace():
                    t_anno = s_anno[offset: offset+len(t)]
                    t_anno = span_anno2single_anno(t,t_anno)
                    if t_anno[0] == 'I':
                        assert prev_t_anno == t_anno or prev_t_anno == t_anno.replace("I-","B-"), f"{token_anno_pairs[-1]}, {(t,t_anno)}"
                    prev_t_anno = t_anno
                    token_anno_pairs.append((t,t_anno))
                offset += len(t)
            all_samples.append(token_anno_pairs)
#             if len(all_samples) % 10000==0:
#                 print(len(all_samples))
    return all_samples

# num_chunks = 1000
# chunk_size = len(labeled_lines)//num_chunks + 1
# all_samples = ray.get([f.remote(labeled_lines[chunk_size*i: min(len(labeled_lines), chunk_size*(i+1))]) for i in range(num_chunks)])
# all_samples = list(chain.from_iterable([x[0] for x in all_processed_data]))
all_samples = f(labeled_lines)

print("Num of sentences: ", len(all_samples))
print("Example: ", all_samples[0])

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T04:31:01.914137Z","iopub.execute_input":"2023-10-31T04:31:01.915122Z","iopub.status.idle":"2023-10-31T04:31:01.923644Z","shell.execute_reply.started":"2023-10-31T04:31:01.915075Z","shell.execute_reply":"2023-10-31T04:31:01.922312Z"}}
print("# of Samples: ", len(all_samples))
print("Example: \n", all_samples[10])
print("Example: \n", all_samples[100])
print("Example: \n", all_samples[1000])
print("Example: \n", all_samples[10000])

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T04:31:01.925711Z","iopub.execute_input":"2023-10-31T04:31:01.926190Z","iopub.status.idle":"2023-10-31T04:31:02.402138Z","shell.execute_reply.started":"2023-10-31T04:31:01.926052Z","shell.execute_reply":"2023-10-31T04:31:02.401066Z"}}
save_path = f'weak.txt'
# with open(save_path, 'w') as f:
#     for s in all_samples:
#         f.write(f"aps\tB-category\n")
#         for t,t_anno in s:
#             f.write(f"{t}\t{t_anno}\n")
#         f.write("\n")
if TGT_ENTITY_TYPE == "Chemical":
    with open("chem_"+save_path, 'w') as f:
        for s in all_samples:
            if not any([e[1]=="B-Chemical" for e in s]):
                continue
            for t,t_anno in s:
                f.write(f'{t}\t{t_anno.replace("B-Disease","O").replace("I-Disease","O")}\n')
            f.write("\n")
if TGT_ENTITY_TYPE == "Disease":
    with open("disease_"+save_path, 'w') as f:
        for s in all_samples:
            if not any([e[1]=="B-Disease" for e in s]):
                continue
            for t,t_anno in s:
                f.write(f'{t}\t{t_anno.replace("B-Chemical","O").replace("I-Chemical","O")}\n')
            f.write("\n")

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code]


# %% [code]
