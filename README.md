# Readme

EECS 595: Natural Language Processing - Final Project

Commonsense in Natural Language Inference (NLI)

## Team 2

* Students: [Ziqiao Ma](https://github.com/Mars-tin), [Xueming Xu](https://github.com/xueming-Felix), [Qingyi Chen](https://github.com/qingyichen)

* Source code: [GitHub](https://github.com/Mars-tin/commonsense-for-inference).

## CommonsenseQA

### Brief Intro

As a question answering benchmark, it presents a natural language question $Q​$ of $m​$ tokens $\{q_1,q_2,\cdots,q_m\}​$ and 5 choices $\{a_1,a_2,\cdots,a_5\}​$ labeled with $\{A,B,\cdots,E\}​$ regarding each question. Notably, the questions do not entail a inference basis in themselves, so the lack of evidence requires the model to hold a comprehensive understanding on common sense knowledge and a strong reasoning ability to make the right choice.

We provided an implementation based on Facebook `fairseq` framework for quicker reproduction. The model is a fine-tuned `roberta` model for the `CommonsenseQA` task .

### Codes

#### Baseline Models

Follow these steps to reproduce the results:

1. Run the `csqa_baseline.ipynb` script by order.

#### `fairseq` Models

Follow these steps to reproduce the results:

1. Unzip the `csqa_fariseq.zip` file for source code;

2. Open the `CommonsenseQA.ipynb` file with Google Colab;

3. Upload the `CommonsenseQA` folder containing all source code to Google Colab;

   In default, the file will be uploaded to `/content/`), as the file is needed for the `.ipynb` script to run;

4. Run the `.ipynb` script by order, the random seed is well fixed and parameters are set as the one we used for the submission.

### Graph Based Model

Follow these steps to reproduce the results:

1. Run the `csqa_graph_reasoning.ipynb` script by order, the random seed is well fixed and parameters are set as the one we used for the submission.
2. If the code crashes, try to down grade the transformers from 2.0 to 1.6.

## Conversation Entailment

### Brief Intro

The Conversation Entailment dataset features a conversation $Q$ composed of $n$ sequences of natural language texts $\{t_1,t_2,\cdots,t_n\}$ as the premise and an interpretation sentence $h$ as the hypothesis. The model is tasked to identify if the hypothesis $h$ is entailed in the given dialogue. The form of conversation and the introduction of different speakers in the dataset distinguishes the dataset from other common textual entailment dataset, and requires the model to reason not only about the facts but also about the speaking subjects on their beliefs, desires, or intentions. 

We provided an implementation based on `huggingface` framework. The model is a fine-tuned `roberta` model for the `EAT` task. The model is first tuned on the `MNLI` dataset for commonsense knowledge source, and then applied on the `ConvEnt` dataset.

We formulate the problem as a binary textual entailment task. The text is preprocessed by joining every part of the conversation into one sentence, as the premise. The hypothesis is attached to the end the premise. Also, the pronoun such as "I" or "you" will be replaced by the corresponding speaker. The further model tuning, training is based on this preprocessed format. The evaluation module will classify the whole sequence.

### Codes

Follow these steps to reproduce the results:

1. Run the `con_vent_mnli.ipynb` script by order, the random seed is well fixed and parameters are set as the one we used for the submission.

## EAT

### Brief Intro

EAT (Everyday Actions in Text) is from the SLED (Situated Language and Embodied Dialogue) group created by Shane Storks. The dataset is in the form of a story of 5 sequential events represented by natural language texts $\{t_1,t_2,\cdots,t_5\}$ respectively. The model aims to identify whether the story is plausible using common sense reasoning and specify at which event the story becomes implausible, if any. Such plausible inference requires the model to have a strong background knowledge and a comprehensive ability to perform common sense reasoning and causal reasoning, since whether an event is plausible in the story is high dependent on the previous events.

We provided an implementation based on `huggingface` framework. The model is a fine-tuned `roberta` model for the `EAT` task. The model is first tuned on the `MNLI` dataset for commonsense knowledge source, and then applied on the `EAT` dataset.

We formulate the problem as a binary textual entailment task. The text is preprocessed by first connecting the sentences before the breakpoint into one sentence for token_1, and the sentence directly after the breakpoint as token_2. The further model tuning, training is based on this preprocessed format. The evaluation module will convert the breakpoint classification result into the plausibility of the whole story and breakpoint location prediction.

### Codes

#### Baselines and Finetuning on `SWAG`

Follow these steps to reproduce the results:

1. Run the `eat_swag.ipynb` script by order, the random seed is well fixed and parameters are set as the one we used for the submission.

#### Finetuning on `MNLI` and `PIQA`

Follow these steps to reproduce the results:

1. Run the `eat_mnli.ipynb` script by order, the random seed is well fixed and parameters are set as the one we used for the submission.
