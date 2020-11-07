import json
import codecs

from datasets import load_dataset


def load_data(dataset='commonsense_qa', preview=5):

    assert dataset in {'commonsense_qa', 'conv_entail', 'eat'}

    if dataset == 'commonsense_qa':
        ds = load_dataset('commonsense_qa')

        if preview > 0:
            data_tr = ds.data['train']
            question = data_tr['question']
            choices = data_tr['choices']
            answerKey = data_tr['answerKey']
            print(question[preview])
            for label, text in zip(choices[preview]['label'], choices[preview]['text']):
                print(label, text)
            print(answerKey[preview])

    elif dataset == 'conv_entail':
        dev_set = codecs.open('data/conv_entail/dev_set.json', 'r', encoding='utf-8').read()
        act_tag = codecs.open('data/conv_entail/act_tag.json', 'r', encoding='utf-8').read()
        ds = json.loads(dev_set), json.loads(act_tag)

        if preview > 0:
            print('Preview not yet implemented for this dataset.')

    else:
        eat = codecs.open('data/eat/eat_train.json', 'r', encoding='utf-8').read()
        ds = json.loads(eat)

        if preview > 0:
            story = ds[preview]['story']
            label = ds[preview]['label']
            bp = ds[preview]['breakpoint']
            for line in story:
                print(line)
            print(label)
            print(bp)

    return ds
