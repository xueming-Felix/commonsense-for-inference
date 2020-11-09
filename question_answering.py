from tqdm import tqdm
import logging

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from utils import load_data

logger = logging.getLogger(__name__)


###########################################
#          Data Processing Classes        #
###########################################
# https://github.com/YOONNAJANG/commonsenseQA/blob/master/run_commonsense_qa_mod.py

class InputExample(object):
    """
    A single multiple choice question.
    """

    def __init__(self, example_id, question, answers, label):
        self.example_id = example_id
        self.question = question
        self.answers = answers
        self.label = label


class InputFeatures(object):
    """
    A single feature converted from an example.
    """

    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.label = label
        self.choices_features = [
            {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids}
            for _, input_ids, input_mask, segment_ids in choices_features
        ]


class CommonsenseQAProcessor:
    """
    A Commonsense QA Data Processor
    """

    def __init__(self):
        self.dataset = None
        self.labels = [0, 1, 2, 3, 4]
        self.LABELS = ['A', 'B', 'C', 'D', 'E']

    def get_split(self, split='train'):
        if self.dataset is None:
            self.dataset = load_data(dataset='commonsense_qa', preview=-1)
        return self.dataset[split]

    def create_examples(self, split='train'):
        examples = []
        data_tr = self.get_split(split)
        example_id = 0

        for question, choices, answerKey in zip(data_tr['question'], data_tr['choices'], data_tr['answerKey']):
            answers = choices['text']
            label = self.LABELS.index(answerKey)
            examples.append(InputExample(
                example_id=example_id, question=question,
                answers=answers, label=label
            ))
            example_id += 1

        return examples


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    Truncates a sequence pair in place to the maximum length.

    This is a simple heuristic which will always truncate the longer sequence one token at a time.
    This makes more sense than truncating an equal percent of tokens from each,
    since if one sequence is very short then each token that's truncated
    likely contains more information than a longer sequence.

    However, since we'd better not to remove tokens of options and questions,
    you can choose to use a bigger length or only pop from context
    """

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            warning = 'Attention! you are removing from token_b (swag task is ok). ' \
                      'If you are training ARC and RACE (you are popping question + options), ' \
                      'you need to try to use a bigger max seq length!'
            logger.info(warning)
            tokens_b.pop()


def examples_to_features(examples, label_list, max_seq_length, tokenizer,
                         cls_token_at_end=False,
                         cls_token='[CLS]',
                         cls_token_segment_id=1,
                         sep_token='[SEP]',
                         sequence_a_segment_id=0,
                         sequence_b_segment_id=1,
                         sep_token_extra=False,
                         pad_token_segment_id=0,
                         pad_on_left=False,
                         pad_token=0,
                         mask_padding_with_zero=True):
    """
    Convert Commonsense QA examples to features.

    The convention in BERT is:
    (a) For sequence pairs:
    tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1

    (b) For single sequences:
    tokens:   [CLS] the dog is hairy . [SEP]
    type_ids:   0   0   0   0  0     0   0

    Where "type_ids" are used to indicate whether this is the first sequence or the second sequence.
    The embedding vectors for `type=0` and `type=1` were learned during pre-training
    and are added to the word piece embedding vector (and position vector).
    This is not *strictly* necessary since the [SEP] token unambiguously separates the sequences,
    but it makes it easier for the model to learn the concept of sequences.

    For classification tasks, the first vector (corresponding to [CLS]) is used as as the "sentence vector".
    Note that this only makes sense because the entire model is fine-tuned.
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples to features"):

        choices_features = []
        for ending_idx, (question, answers) in enumerate(zip(example.question, example.answers)):

            tokens_a = tokenizer.tokenize(example.question)
            if example.question.find("_") != -1:
                tokens_b = tokenizer.tokenize(example.question.replace("_", answers))
            else:
                tokens_b = tokenizer.tokenize(answers)

            special_tokens_count = 4 if sep_token_extra else 3
            truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)

            tokens = tokens_a + [sep_token]
            if sep_token_extra:
                tokens += [sep_token]

            segment_ids = [sequence_a_segment_id] * len(tokens)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens.
            # Only real tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)

            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids

            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("tokens: {}".format(' '.join(tokens)))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
                logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
                logger.info("label: {}".format(label))

        features.append(InputFeatures(
            example_id=example.example_id,
            choices_features=choices_features,
            label=label
        ))

    return features


def load_dataset(args, tokenizer, mode='train'):
    """
    Load the processed Commonsense QA dataset
    """

    def select_field(feature_list, field_name):
        return [
            [choice[field_name] for choice in feature.choices_features]
            for feature in feature_list
        ]

    assert mode in {'train', 'validation', 'test'}
    logger.info("Creating features from dataset...")

    processor = CommonsenseQAProcessor()
    label_list = processor.labels
    examples = processor.create_examples(split=mode)

    logger.info("Training number: %s", str(len(examples)))
    features = examples_to_features(examples, label_list, args.max_seq_length, tokenizer,
                                    cls_token_at_end=bool(args.model_type in ['xlnet']),
                                    cls_token=tokenizer.cls_token,
                                    sep_token=tokenizer.sep_token,
                                    sep_token_extra=bool(args.model_type in ['roberta']),
                                    cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                    pad_on_left=bool(args.model_type in ['xlnet']),
                                    pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


###########################################
#           Deep Neural Networks          #
###########################################


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = self.softmax(tag_space)
        return tag_scores


class BiLstm(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim,
                 n_layers, bidirectional, dropout, pad_idx):
        super(BiLstm, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.apply(self.init_weights)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        outputs, _ = self.lstm(embedded)
        predictions = self.fc(self.dropout(outputs))
        return predictions

    @staticmethod
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.normal_(param.data, mean=0, std=1e-1)


class Bert(nn.Module):
    def __init__(self, bert, output_dim, dropout):
        super(Bert, self).__init__()

        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        text = text.permute(1, 0)
        embedded = self.dropout(self.bert(text)[0])
        embedded = embedded.permute(1, 0, 2)
        predictions = self.fc(self.dropout(embedded))
        return predictions
