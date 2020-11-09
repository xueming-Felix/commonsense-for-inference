import argparse
import torch

from question_answering import load_dataset, CommonsenseQAProcessor
from utils import set_seed, load_model


def train(args, dataset, model, tokenizer):

    return model


def test():
    raise NotImplemented


def main(args):
    set_seed(args.seed)

    processor = CommonsenseQAProcessor()
    num_labels = len(processor.labels)

    config_class, model_class, tokenizer_class = load_model(args.model_type)
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name,
        num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name,
        do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(
        args.model_name, from_tf=bool('.ckpt' in args.model_name), config=config)

    model.to(args.device)

    train_set = load_dataset(args, tokenizer, mode='train')

    if args.mode == 'train':
        model = train(args, train_set, model, tokenizer)
        # torch.save(model, args.save_file)
    else:
        test()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Common sense question answering")

    parser.add_argument("--mode", type=str, default='train',
                        help="Mode: <str> [ train | test ]")
    parser.add_argument("--model_type", type=str, default='bert',
                        help="Model: <str> [ bert | xlnet | roberta | gpt2 ]")
    parser.add_argument("--task_name", default=None, type=str, required=False,
                        help="The name of the task to train: <str> [ commonqa ]")
    parser.add_argument("--model_name", type=str, required=True,
                        default='/mnt/raid5/shared/bert/tensorflow/cased_L-12_H-768_A-12/bert_config.json',
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", "
                        .join(load_model('all')))
    parser.add_argument("--config_name", type=str,
                        default="/mnt/raid5/shared/bert/tensorflow/cased_L-12_H-768_A-12/bert_config.json",
                        help="Pre-trained config name or path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pre-trained tokenizer name or path if not the same as model_name")

    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--seed", type=int, default=0, help="Random seed: <int>")
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    main(parser.parse_args())
