#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main OpenQA training and testing script."""

import argparse
import torch
import numpy as np
import json
import os
import sys
import subprocess
import logging
import random
import pickle

import regex as re

sys_dir = '/home/niuyilin/OpenQA-STM'
sys.path.append(sys_dir)

from src.reader import utils, vector, config, data
from src.reader import DocReader
from src import DATA_DIR as DRQA_DATA
from src.retriever.utils import normalize
from src.reader.data import Dictionary


from src import tokenizers
from multiprocessing.util import Finalize
tokenizers.set_default('corenlp_classpath', sys_dir+'/data/corenlp/*')
PROCESS_TOK = None



logger = logging.getLogger()


# ------------------------------------------------------------------------------
# Training arguments.
# ------------------------------------------------------------------------------


# Defaults
DATA_DIR = os.path.join(DRQA_DATA, 'datasets')
MODEL_DIR = 'models' 
EMBED_DIR = sys_dir+'/data/embeddings/' 

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--dataset', type=str, default="searchqa",
                         help='Dataset: searchqa, quasart or unftriviaqa')

    runtime.add_argument('--mode', type=str, default="all",
                         help='Train_mode: all, reader or selector')
    runtime.add_argument('--no-cuda', type='bool', default=False,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help='Run on a specific GPU')
    runtime.add_argument('--data-workers', type=int, default=1,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--parallel', type='bool', default=False,
                         help='Use DataParallel on all available GPUs')
    runtime.add_argument('--random-seed', type=int, default=1012,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num-epochs', type=int, default=20,
                         help='Train data iterations')
    runtime.add_argument('--batch-size', type=int, default=128,
                         help='Batch size for training')
    runtime.add_argument('--test-batch-size', type=int, default=64,
                         help='Batch size during validation/testing')

    # Co-Training
    cotraining = parser.add_argument_group('CoTraining')
    cotraining.add_argument('--top_k', type=int, default=2000,
                         help='Number of labeling samples in Co-Training')
    cotraining.add_argument('--load_evidence_file', type=str, default='none',
                         help='Path of sentence id file')
    cotraining.add_argument('--save_evidence_file', type=str, default='none',
                         help='Path of sentence id file')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model-dir', type=str, default=MODEL_DIR,
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model-name', type=str, default='SQuAD.ckpt_tmp',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data-dir', type=str, default=DATA_DIR,
                       help='Directory of training/validation data')
    files.add_argument('--embed-dir', type=str, default=EMBED_DIR,
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding-file', type=str,
                       default='glove.840B.300d.txt',
                       help='Space-separated pretrained embeddings file')

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default= None, #'models/SQuAD.ckpt.mdl',#'data/reader/multitask.mdl
                           help='Path to a pretrained model to warm-start with')
    save_load.add_argument('--expand-dictionary', type='bool', default=False,
                           help='Expand dictionary of pretrained model to ' +
                                'include training/dev words of new data')
    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--uncased-question', type='bool', default=False,
                            help='Question words will be lower-cased')
    preprocess.add_argument('--uncased-doc', type='bool', default=False,
                            help='Document words will be lower-cased')
    preprocess.add_argument('--restrict-vocab', type='bool', default=True,
                            help='Only use pre-trained words in embedding_file')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--official-eval', type='bool', default=True,
                         help='Validate with official SQuAD eval')
    general.add_argument('--valid-metric', type=str, default='exact_match',
            help='If using official evaluation: f1; else: exact_match')
    general.add_argument('--display-iter', type=int, default=25,
                         help='Log state after every <display_iter> epochs')
    general.add_argument('--sort-by-len', type='bool', default=True,
                         help='Sort batches by length for speed')


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    if args.embedding_file:
        args.embedding_file = os.path.join(args.embed_dir, args.embedding_file)
        if not os.path.isfile(args.embedding_file):
            raise IOError('No such file: %s' % args.embedding_file)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')

    # Embeddings options
    if args.embedding_file:
        with open(args.embedding_file) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        args.embedding_dim = dim
    elif not args.embedding_dim:
        raise RuntimeError('Either embedding_file or embedding_dim '
                           'needs to be specified.')

    # Make sure tune_partial and fix_embeddings are consistent.
    if args.tune_partial > 0 and args.fix_embeddings:
        logger.warning('WARN: fix_embeddings set to False as tune_partial > 0.')
        args.fix_embeddings = False

    # Make sure fix_embeddings and embedding_file are consistent
    if args.fix_embeddings:
        if not (args.embedding_file or args.pretrained):
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False
    return args


# ------------------------------------------------------------------------------
# Initalization from scratch.
# ------------------------------------------------------------------------------


def init_from_scratch(args, train_docs):
    """New model, new data, new dictionary."""
    # Create a feature dict out of the annotations in the data
    logger.info('-' * 100)
    logger.info('Generate features')
    feature_dict = utils.build_feature_dict(args)
    logger.info('Num features = %d' % len(feature_dict))
    logger.info(feature_dict)

    # Build a dictionary from the data questions + words (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build dictionary')
    word_dict = utils.build_word_dict_docs(args, train_docs)

    logger.info('Num words = %d' % len(word_dict))

    # Initialize model
    model = DocReader(config.get_model_args(args), word_dict, feature_dict)

    # Load pretrained embeddings for words in dictionary
    if args.embedding_file:
        model.load_embeddings(word_dict.tokens(), args.embedding_file)

    return model


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------

def train(args, data_loader, model, global_stats, exs_with_doc, docs_by_question):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()
    # Run one epoch
    update_step = 0
    for idx, ex_with_doc in enumerate(data_loader):
        ex = ex_with_doc[0]
        batch_size, question, ex_id = ex[0].size(0), ex[3], ex[-1]
        if (idx not in HasAnswer_Map):
            HasAnswer_list = []
            for idx_doc in range(0, vector.num_docs):
                HasAnswer = []
                for i in range(batch_size):
                    HasAnswer.append(has_answer(args, exs_with_doc[ex_id[i]]['answer'], docs_by_question[ex_id[i]][idx_doc%len(docs_by_question[ex_id[i]])]["document"]))
                HasAnswer_list.append(HasAnswer)
            HasAnswer_Map[idx] = HasAnswer_list
        else:
            HasAnswer_list = HasAnswer_Map[idx]

        if (idx not in Evidence_Label):
            Evidence_list = [-1] * batch_size
            Evidence_Label[idx] = Evidence_list
        else:
            Evidence_list = Evidence_Label[idx]

        weights = []
        for idx_doc in range(0, vector.num_docs):
            weights.append(1)
        weights = torch.Tensor(weights)
        idx_random = torch.multinomial(weights, int(vector.num_docs))
        idx_doc_map = {int(idx_doc): i for i, idx_doc in enumerate(idx_random)}
        idx_doc_map[-1] = -1

        HasAnswer_list_sample = []
        Evidence_list_sample = []
        ex_with_doc_sample = []
        for i, idx_doc in enumerate(idx_random):
            HasAnswer_list_sample.append(HasAnswer_list[idx_doc])
            ex_with_doc_sample.append(ex_with_doc[idx_doc])
        for i in range(batch_size):
            Evidence_list_sample.append(idx_doc_map[Evidence_list[i]])

        l_list_doc = []
        r_list_doc = []
        for idx_doc in idx_random:
            l_list = []
            r_list = []
            for i in range(batch_size):
                if HasAnswer_list[idx_doc][i][0]:
                    l_list.append(HasAnswer_list[idx_doc][i][1])
                else:
                    l_list.append((-1,-1))
            l_list_doc.append(l_list)
            r_list_doc.append(r_list)
        pred_s_list_doc = []
        pred_e_list_doc = []
        tmp_top_n = 1
        for idx_doc in idx_random:
            ex = ex_with_doc[idx_doc]
            pred_s, pred_e, pred_score = model.predict(ex,top_n = tmp_top_n)
            pred_s_list = []
            pred_e_list = []
            for i in range(batch_size):
                pred_s_list.append(pred_s[i].tolist())
                pred_e_list.append(pred_e[i].tolist())
            pred_s_list_doc.append(torch.LongTensor(pred_s_list))
            pred_e_list_doc.append(torch.LongTensor(pred_e_list))

        _loss = model.update_with_doc(update_step, ex_with_doc_sample, \
                            pred_s_list_doc, pred_e_list_doc, tmp_top_n, \
                            l_list_doc, r_list_doc, HasAnswer_list_sample, \
                            evidence_label=Evidence_list_sample)
        train_loss.update(*_loss)
        update_step = (update_step + 1) % 4
        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'loss = %.2f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()
        if (idx%200==199):
            validate_unofficial_with_doc(args, data_loader, model, global_stats, exs_with_doc, docs_by_question, 'train')
        # break
    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint',
                         global_stats['epoch'] + 1)


def update_evidence(args, data_loader, model, global_stats, exs_with_doc, docs_by_question):
    Top_k = args.top_k
    logger.info('Top k is set to %d' % (Top_k))

    Probability = {}
    Attention_Weight = {}

    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_prob = utils.AverageMeter()
    train_attention = utils.AverageMeter()
    epoch_time = utils.Timer()
    # Run one epoch
    update_step = 0
    for idx, ex_with_doc in enumerate(data_loader):
        ex = ex_with_doc[0]
        batch_size, question, ex_id = ex[0].size(0), ex[3], ex[-1]
        if (idx not in HasAnswer_Map):
            HasAnswer_list = []
            for idx_doc in range(0, vector.num_docs):
                HasAnswer = []
                for i in range(batch_size):
                    HasAnswer.append(has_answer(args, exs_with_doc[ex_id[i]]['answer'], docs_by_question[ex_id[i]][idx_doc%len(docs_by_question[ex_id[i]])]["document"]))
                HasAnswer_list.append(HasAnswer)
            HasAnswer_Map[idx] = HasAnswer_list
        else:
            HasAnswer_list = HasAnswer_Map[idx]

        if (idx not in Evidence_Label):
            Evidence_list = [-1] * batch_size
            Evidence_Label[idx] = Evidence_list

        # Don't shuffle when update evidence
        idx_random = range(vector.num_docs)

        HasAnswer_list_sample = []
        ex_with_doc_sample = []
        for idx_doc in idx_random:
            HasAnswer_list_sample.append(HasAnswer_list[idx_doc])
            ex_with_doc_sample.append(ex_with_doc[idx_doc])

        l_list_doc = []
        r_list_doc = []
        for idx_doc in idx_random:
            l_list = []
            r_list = []
            for i in range(batch_size):
                if HasAnswer_list[idx_doc][i][0]:
                    l_list.append(HasAnswer_list[idx_doc][i][1])
                else:
                    l_list.append((-1,-1))
            l_list_doc.append(l_list)
            r_list_doc.append(r_list)
        pred_s_list_doc = []
        pred_e_list_doc = []
        tmp_top_n = 1
        for idx_doc in idx_random:
            ex = ex_with_doc[idx_doc]
            pred_s, pred_e, pred_score = model.predict(ex,top_n = tmp_top_n)
            pred_s_list = []
            pred_e_list = []
            for i in range(batch_size):
                pred_s_list.append(pred_s[i].tolist())
                pred_e_list.append(pred_e[i].tolist())
            pred_s_list_doc.append(torch.LongTensor(pred_s_list))
            pred_e_list_doc.append(torch.LongTensor(pred_e_list))

        probs, attentions = model.update_with_doc(update_step, ex_with_doc_sample, \
                                        pred_s_list_doc, pred_e_list_doc, tmp_top_n, \
                                        l_list_doc, r_list_doc, HasAnswer_list_sample, \
                                        return_prob=True)
        train_prob.update(np.mean(probs), batch_size)
        train_attention.update(np.mean(attentions[0]), batch_size)
        update_step = (update_step + 1) % 4

        if idx % args.display_iter == 0:
            logger.info('Update Evidence: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'Average prob = %f | Average attention = %f | elapsed time = %.2f (s)' %
                        (train_prob.avg, train_attention.avg, global_stats['timer'].time()))

        for i in range(batch_size):
            key = "%d|%d" % (idx, i)
            if key in Probability or key in Attention_Weight:
                raise ValueError("%s exists in Probability or Attention_Weight" % (key))
            # Add threshold here
            Probability[key] = probs[i]
            Attention_Weight[key] = (attentions[0][i], attentions[1][i]) # max_value, max_index
        # break
    evidence_scores = {key: (attention[0], attention[1]) for key, attention in Attention_Weight.items() if attention[1] != -1}
    evidence_scores = sorted(evidence_scores.items(), key=lambda x: x[1][0], reverse=True)
    count = 0
    label_prob = []
    label_attention = []
    for key, value in evidence_scores:
        idx, i = key.split('|')
        idx = int(idx)
        i = int(i)
        if Evidence_Label[idx][i] != -1:
            continue
        count += 1
        Evidence_Label[idx][i] = value[1]
        label_prob.append(Probability[key])
        label_attention.append(Attention_Weight[key][0])
        if count >= Top_k:
            break

    logger.info('Update Evidence: Epoch %d done. Time for epoch = %.2f (s). Average prob = %f. Average attention = %f.' %
                (global_stats['epoch'], epoch_time.time(), train_prob.avg, train_attention.avg))
    logger.info('Update Evidence: Label %d examples. Average prob = %f. Average attention = %f.' %
                (count, np.mean(label_prob), np.mean(label_attention)))

HasAnswer_Map = {}
Evidence_Label = {}
def pretrain_selector(args, data_loader, model, global_stats, exs_with_doc, docs_by_question):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()
    # Run one epoch
    tot_ans = 0
    tot_num = 0
    global HasAnswer_Map
    for idx, ex_with_doc in enumerate(data_loader):
        ex = ex_with_doc[0]
        batch_size, question, ex_id = ex[0].size(0), ex[3], ex[-1]
        if (idx not in HasAnswer_Map):
            HasAnswer_list = []
            for idx_doc in range(0, vector.num_docs):
                HasAnswer = []
                for i in range(batch_size):
                    has_a, a_l = has_answer(args, exs_with_doc[ex_id[i]]['answer'], docs_by_question[ex_id[i]][idx_doc%len(docs_by_question[ex_id[i]])]["document"])
                    HasAnswer.append(has_a)
                HasAnswer_list.append(HasAnswer)
            #HasAnswer_list = torch.LongTensor(HasAnswer_list)
            HasAnswer_Map[idx] = HasAnswer_list
        else:
            HasAnswer_list = HasAnswer_Map[idx]
        for idx_doc in range(0, vector.num_docs):
            for i in range(batch_size):
                tot_ans+=HasAnswer_list[idx_doc][i]
                tot_num+=1

        weights = []
        for idx_doc in range(0, vector.num_docs):
            weights.append(1)
        weights = torch.Tensor(weights)
        idx_random = torch.multinomial(weights, int(vector.num_docs))

        HasAnswer_list_sample = []
        ex_with_doc_sample = []
        for idx_doc in idx_random:
            HasAnswer_list_sample.append(HasAnswer_list[idx_doc])
            ex_with_doc_sample.append(ex_with_doc[idx_doc])
        HasAnswer_list_sample = torch.LongTensor(HasAnswer_list_sample)

        train_loss.update(*model.pretrain_selector(ex_with_doc_sample, HasAnswer_list_sample))
        #train_loss.update(*model.pretrain_ranker(ex_with_doc, HasAnswer_list))
        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'loss = %.2f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            logger.info("tot_ans:\t%d\t%d\t%f", tot_ans, tot_num, tot_ans*1.0/tot_num)
            train_loss.reset()
    logger.info("tot_ans:\t%d\t%d", tot_ans, tot_num)
    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

def pretrain_reader(args, data_loader, model, global_stats, exs_with_doc, docs_by_question):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()
    logger.info("pretrain_reader")
    # Run one epoch
    global HasAnswer_Map
    count_ans = 0
    count_tot = 0
    for idx, ex_with_doc in enumerate(data_loader):
        #logger.info(idx)
        ex = ex_with_doc[0]
        batch_size, question, ex_id = ex[0].size(0), ex[3], ex[-1]
        if (idx not in HasAnswer_Map):
            HasAnswer_list = []
            for idx_doc in range(0, vector.num_docs):
                HasAnswer = []
                for i in range(batch_size):
                    HasAnswer.append(has_answer(args,exs_with_doc[ex_id[i]]['answer'], docs_by_question[ex_id[i]][idx_doc%len(docs_by_question[ex_id[i]])]["document"]))
                HasAnswer_list.append(HasAnswer)
            HasAnswer_Map[idx] = HasAnswer_list
        else:
            HasAnswer_list = HasAnswer_Map[idx]
       
        for idx_doc in range(0, vector.num_docs):
            l_list = []
            r_list = []
            pred_s, pred_e, pred_score = model.predict(ex_with_doc[idx_doc],top_n = 1)
            for i in range(batch_size):
                if HasAnswer_list[idx_doc][i][0]:
                    count_ans+=len(HasAnswer_list[idx_doc][i][1])
                    count_tot+=1
                    l_list.append(HasAnswer_list[idx_doc][i][1])
                else:
                    l_list.append([(int(pred_s[i][0]),int(pred_e[i][0]))])
            train_loss.update(*model.update(ex_with_doc[idx_doc], l_list, r_list, HasAnswer_list[idx_doc])) 
        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'loss = %.2f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()
            logger.info("%d\t%d\t%f", count_ans, count_tot, 1.0*count_ans/(count_tot+1))
    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

def has_answer(args, answer, t):
    global PROCESS_TOK
    text = []
    for i in range(len(t)):
        text.append(t[i].lower())
    res_list = []
    if (args.dataset == "CuratedTrec"):
        try:
            ans_regex = re.compile("(%s)"%answer[0], flags=re.IGNORECASE + re.UNICODE)
        except:
            return False, res_list
        paragraph = " ".join(text)
        answer_new = ans_regex.findall(paragraph)
        for a in answer_new:
            single_answer = normalize(a[0])
            single_answer = PROCESS_TOK.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    res_list.append((i, i+len(single_answer)-1))
    else:
        for a in answer:
            single_answer = " ".join(a).lower()
            single_answer = normalize(single_answer)
            single_answer = PROCESS_TOK.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    res_list.append((i, i+len(single_answer)-1))
    if (len(res_list)>0):
        return True, res_list
    else:
        return False, res_list


def set_sim(answer, prediction):

    ground_truths = []
    for a in answer:
        ground_truths.append(" ".join([w for w in a]))

    res = utils.metric_max_over_ground_truths(
                utils.f1_score, prediction, ground_truths)
    return res


# ------------------------------------------------------------------------------
# Validation loops. Includes both "unofficial" and "official" functions that
# use different metrics and implementations.
# ------------------------------------------------------------------------------




def validate_unofficial_with_doc(args, data_loader, model, global_stats, exs_with_doc, docs_by_question, mode):
    """Run one full unofficial validation with docs.
    Unofficial = doesn't use SQuAD script.
    """
    eval_time = utils.Timer()
    f1 = utils.AverageMeter()
    exact_match = utils.AverageMeter()

    out_set = set({33,42,45,70,39})
    logger.info("validate_unofficial_with_doc")
    # Run through examples

    examples = 0       
    aa = [0.0 for i in range(vector.num_docs)]
    bb = [0.0 for i in range(vector.num_docs)]
    aa_sum = 0.0
    display_num = 10
    for idx, ex_with_doc in enumerate(data_loader):
        ex = ex_with_doc[0]
        batch_size, question, ex_id = ex[0].size(0), ex[3], ex[-1]
        scores_doc_num = model.predict_with_doc(ex_with_doc)
        scores = [{} for i in range(batch_size)]

        tot_sum = [0.0 for i in range(batch_size)]
        tot_sum1 = [0.0 for i in range(batch_size)]
        neg_sum = [0.0 for i in range(batch_size)]
        min_sum = [[] for i in range(batch_size)]
        min_sum1 =[[] for i in range(batch_size)]
        
        for idx_doc in range(0, vector.num_docs):
            ex = ex_with_doc[idx_doc]
            pred_s, pred_e, pred_score = model.predict(ex,top_n = 10)
            for i in range(batch_size):
                doc_text = docs_by_question[ex_id[i]][idx_doc%len(docs_by_question[ex_id[i]])]["document"]
                has_answer_t = has_answer(args, exs_with_doc[ex_id[i]]['answer'], doc_text)

                for k in range(10):
                    try:
                        prediction = []
                        for j in range(pred_s[i][k], pred_e[i][k]+1):
                            prediction.append(doc_text[j])
                        prediction = " ".join(prediction).lower()
                        if (prediction not in scores[i]):
                            scores[i][prediction] = 0
                        scores[i][prediction] += pred_score[i][k]*scores_doc_num[i][idx_doc]
                    except:
                        pass
        for i in range(batch_size):
            _, indices = scores_doc_num[i].sort(0, descending = True)
            for j in range(0, display_num):
                idx_doc = indices[j]
                doc_text = docs_by_question[ex_id[i]][idx_doc%len(docs_by_question[ex_id[i]])]["document"]
                if (has_answer(args, exs_with_doc[ex_id[i]]['answer'], doc_text)[0]):

                    aa[j]= aa[j] + 1
                bb[j]= bb[j]+1

        for i in range(batch_size):
            
            best_score = 0
            prediction = ""
            for key in scores[i]:
                if (scores[i][key]>best_score):
                    best_score = scores[i][key]
                    prediction = key
            
            # Compute metrics
            ground_truths = []
            answer = exs_with_doc[ex_id[i]]['answer']
            if (args.dataset == "CuratedTrec"):
                ground_truths = answer
            else:
                for a in answer:
                    ground_truths.append(" ".join([w for w in a]))
            #logger.info(prediction)
            #logger.info(ground_truths)
            exact_match.update(utils.metric_max_over_ground_truths(
                utils.exact_match_score, prediction, ground_truths))
            f1.update(utils.metric_max_over_ground_truths(
                utils.f1_score, prediction, ground_truths))
            a = sorted(scores[i].items(), key=lambda d: d[1], reverse = True) 

        examples += batch_size
        if (mode=="train" and examples>=1000):
            break
    try:
        for j in range(0, display_num):
            if (j>0):
                aa[j]= aa[j]+aa[j-1]
                bb[j]= bb[j]+bb[j-1]
            logger.info(aa[j]/bb[j])
    except:
        pass
    logger.info('%s valid official with doc: Epoch = %d | EM = %.2f | ' %
                (mode, global_stats['epoch'], exact_match.avg * 100) +
                'F1 = %.2f | examples = %d | valid time = %.2f (s)' %
                (f1.avg * 100, examples, eval_time.time()))

    return {'exact_match': exact_match.avg * 100, 'f1': f1.avg * 100}


def eval_accuracies(pred_s, target_s, pred_e, target_e):
    """An unofficial evalutation helper.
    Compute exact start/end/complete match accuracies for a batch.
    """
    # Convert 1D tensors to lists of lists (compatibility)
    if torch.is_tensor(target_s):
        target_s = [[e] for e in target_s]
        target_e = [[e] for e in target_e]

    # Compute accuracies from targets
    batch_size = len(pred_s)
    start = utils.AverageMeter()
    end = utils.AverageMeter()
    em = utils.AverageMeter()
    for i in range(batch_size):
        # Start matches
        if pred_s[i] in target_s[i]:
            start.update(1)
        else:
            start.update(0)

        # End matches
        if pred_e[i] in target_e[i]:
            end.update(1)
        else:
            end.update(0)

        # Both start and end match
        if any([1 for _s, _e in zip(target_s[i], target_e[i])
                if _s == pred_s[i] and _e == pred_e[i]]):
            em.update(1)
        else:
            em.update(0)
    return start.avg * 100, end.avg * 100, em.avg * 100


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def read_data(filename, keys):
    res = []
    step = 0
    for line in open(filename):
        data = json.loads(line)
        if ('squad' in filename or 'webquestions' in filename):
            answer = [tokenize_text(a).words() for a in data['answer']]
        else:
            if ('CuratedTrec' in filename):
                answer = data['answer']
            else:
                answer = [tokenize_text(a).words() for a in data['answers']]
        question = " ".join(tokenize_text(data['question']).words())
        res.append({"answer":answer, "question":question})
        step+=1
    return res
    

def tokenize_text(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)

def main(args):
    # --------------------------------------------------------------------------
    # TOK
    global PROCESS_TOK
    tok_class = tokenizers.get_class("corenlp")
    tok_opts = {}
    PROCESS_TOK = tok_class(**tok_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)

    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    dataset = args.dataset#'quasart'#'searchqa'#'unftriviaqa'#'squad'#
    filename_train_docs = sys_dir+"/data/datasets/"+dataset+"/train.json" 
    filename_dev_docs = sys_dir+"/data/datasets/"+dataset+"/dev.json" 
    filename_test_docs = sys_dir+"/data/datasets/"+dataset+"/test.json" 
    train_docs, train_questions = utils.load_data_with_doc(args, filename_train_docs)
    logger.info(len(train_docs))
    filename_train = sys_dir+"/data/datasets/"+dataset+"/train.txt" 
    filename_dev = sys_dir+"/data/datasets/"+dataset+"/dev.txt" 
    train_exs_with_doc = read_data(filename_train, train_questions)

    logger.info('Num train examples = %d' % len(train_exs_with_doc))

    dev_docs, dev_questions = utils.load_data_with_doc(args, filename_dev_docs)
    logger.info(len(dev_docs))
    dev_exs_with_doc = read_data(filename_dev, dev_questions)
    logger.info('Num dev examples = %d' % len(dev_exs_with_doc))

    test_docs, test_questions = utils.load_data_with_doc(args, filename_test_docs)
    logger.info(len(test_docs))
    test_exs_with_doc = read_data(sys_dir+"/data/datasets/"+dataset+"/test.txt", test_questions)
    logger.info('Num dev examples = %d' % len(test_exs_with_doc))
  
    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 0
    if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
        # Just resume training, no modifications.
        logger.info('Found a checkpoint...')
        checkpoint_file = args.model_file + '.checkpoint'
        model, start_epoch = DocReader.load_checkpoint(checkpoint_file)
        #model = DocReader.load(checkpoint_file, args)
        start_epoch = 0
    else:
        # Training starts fresh. But the model state is either pretrained or
        # newly (randomly) initialized.
        if args.pretrained:
            logger.info('Using pretrained model...')
            model = DocReader.load(args.pretrained, args)
            if args.expand_dictionary:
                logger.info('Expanding dictionary for new data...')
                # Add words in training + dev examples
                words = utils.load_words(args, train_exs + dev_exs)
                added = model.expand_dictionary(words)
                # Load pretrained embeddings for added words
                if args.embedding_file:
                    model.load_embeddings(added, args.embedding_file)

        else:
            logger.info('Training model from scratch...')
            model = init_from_scratch(args, train_docs)#, train_exs, dev_exs)

        # Set up optimizer
        model.init_optimizer()

    # Use the GPU?
    if args.cuda:
        model.cuda()

    # Use multiple GPUs?
    if args.parallel:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')


    train_dataset_with_doc = data.ReaderDataset_with_Doc(train_exs_with_doc, model, train_docs, single_answer=True)
    train_sampler_with_doc = torch.utils.data.sampler.SequentialSampler(train_dataset_with_doc)
    train_loader_with_doc = torch.utils.data.DataLoader(
        train_dataset_with_doc,
        batch_size=args.batch_size,
        sampler=train_sampler_with_doc,
        num_workers=args.data_workers,
        collate_fn=vector.batchify_with_docs,
        pin_memory=args.cuda,
    )

    dev_dataset_with_doc = data.ReaderDataset_with_Doc(dev_exs_with_doc, model, dev_docs, single_answer=False)
    dev_sampler_with_doc = torch.utils.data.sampler.SequentialSampler(dev_dataset_with_doc)
    dev_loader_with_doc = torch.utils.data.DataLoader(
        dev_dataset_with_doc,
        batch_size=args.test_batch_size,
        sampler=dev_sampler_with_doc,
        num_workers=args.data_workers,
        collate_fn=vector.batchify_with_docs,
        pin_memory=args.cuda,
    )

    test_dataset_with_doc = data.ReaderDataset_with_Doc(test_exs_with_doc, model, test_docs, single_answer=False)
    test_sampler_with_doc = torch.utils.data.sampler.SequentialSampler(test_dataset_with_doc)
    test_loader_with_doc = torch.utils.data.DataLoader(
       test_dataset_with_doc,
       batch_size=args.test_batch_size,
       sampler=test_sampler_with_doc,
       num_workers=args.data_workers,
       collate_fn=vector.batchify_with_docs,
       pin_memory=args.cuda,
    )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    logger.info('-' * 100)
    logger.info('Starting training...')
    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}

          
    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch

        # Train
        if (args.mode == 'all'):
            train(args, train_loader_with_doc, model, stats, train_exs_with_doc, train_docs)
        if (args.mode == 'reader'):
            pretrain_reader(args, train_loader_with_doc, model, stats, train_exs_with_doc, train_docs)
        if (args.mode == 'selector'):
            pretrain_selector(args, train_loader_with_doc, model, stats, train_exs_with_doc, train_docs)
        
        result = validate_unofficial_with_doc(args, dev_loader_with_doc, model, stats, dev_exs_with_doc, dev_docs, 'dev')
        validate_unofficial_with_doc(args, train_loader_with_doc, model, stats, train_exs_with_doc, train_docs, 'train')
        if (dataset=='webquestions' or dataset=='CuratedTrec'):
            result = validate_unofficial_with_doc(args, test_loader_with_doc, model, stats, test_exs_with_doc, test_docs, 'test')
        else:
            validate_unofficial_with_doc(args, test_loader_with_doc, model, stats, test_exs_with_doc, test_docs, 'test')
        if result[args.valid_metric] > stats['best_valid']:
            logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                        (args.valid_metric, result[args.valid_metric],
                         stats['epoch'], model.updates))
            model.save(args.model_file)

            stats['best_valid'] = result[args.valid_metric]

    #Update evidence label
    if args.save_evidence_file != 'none':
        model.load(args.model_file)
        update_evidence(args, train_loader_with_doc, model, stats, train_exs_with_doc, train_docs)
        pickle.dump(Evidence_Label, open(os.path.join(args.model_dir, args.model_name + '.%s.pkl' % (args.save_evidence_file)), 'wb'))

def split_doc(doc):
    """Given a doc, split it into chunks (by paragraph)."""
    GROUP_LENGTH = 0
    curr = []
    curr_len = 0
    for split in regex.split(r'\n+', doc):
        split = split.strip()
        if len(split) == 0:
            continue
        # Maybe group paragraphs together until we hit a length limit
        if len(curr) > 0 and curr_len + len(split) > GROUP_LENGTH:
            yield ' '.join(curr)
            curr = []
            curr_len = 0
        curr.append(split)
        curr_len += len(split)
    if len(curr) > 0:
        yield ' '.join(curr)


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'DrQA Document Reader',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    if args.load_evidence_file != 'none':
        Evidence_Label = pickle.load(open(os.path.join(args.model_dir, args.model_name + '.%s.pkl' % (args.load_evidence_file)), 'rb'))

    # Set cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)
