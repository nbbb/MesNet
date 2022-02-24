import inspect
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys
import pandas as pd
sys.path.append('coco-caption')
import losses
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from torchvision import transforms
from loader.transform import  TrimExceptAscii, Lowercase, \
                            RemovePunctuation, SplitWithWhiteSpace, Truncate
from configs.train_stage1 import TESTConfig as TEST
import json
class LossChecker:
    def __init__(self, num_losses):
        self.num_losses = num_losses

        self.losses = [ [] for _ in range(self.num_losses) ]

    def update(self, *loss_vals):
        assert len(loss_vals) == self.num_losses

        for i, loss_val in enumerate(loss_vals):
            self.losses[i].append(loss_val)

    def mean(self, last=0):
        mean_losses = [ 0. for _ in range(self.num_losses) ]
        for i, loss in enumerate(self.losses):
            _loss = loss[-last:]
            mean_losses[i] = sum(_loss) / len(_loss)
        return mean_losses


def parse_batch(batch):
    vids, feats, captions = batch
    feats = [ feat.cuda() for feat in feats ]
    feats = torch.cat(feats, dim=2)
    captions = captions.long().cuda()
    return vids, feats, captions
def train(e, model, optimizer, train_iter, vocab, teacher_forcing_ratio, reg_lambda, recon_lambda, gradient_clip):
    model.train()
    loss_checker = LossChecker(3)
    PAD_idx = vocab.word2idx['<PAD>']
    # b_parms=balance_param(vocab)

    t = tqdm(train_iter)
    for batch in t:
        _, feats, captions = parse_batch(batch)
        optimizer.zero_grad()
        output = model(feats, captions, teacher_forcing_ratio)

        captions = captions.transpose(0, 1).contiguous()  # 变回原维度
        output = output.transpose(0, 1).contiguous()  # 输出维度变回原维度
        cross_entropy_loss = F.nll_loss(output[1:].view(-1, vocab.n_vocabs),
                                        captions[1:].contiguous().view(-1),
                                        ignore_index=PAD_idx,
                                        )#weight=b_parms
        entropy_loss = losses.entropy_loss(output[1:], ignore_mask=(captions[1:] == PAD_idx))
        loss = cross_entropy_loss + reg_lambda * entropy_loss


        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        loss_checker.update(loss.item(), cross_entropy_loss.item(), entropy_loss.item())
        t.set_description("[Epoch #{0}] loss: {3:.3f} = (CE: {4:.3f}) + (Ent: {1} * {5:.3f}) ".format(e, reg_lambda, recon_lambda, *loss_checker.mean(last=10)))
    total_loss, cross_entropy_loss, entropy_loss = loss_checker.mean()
    loss = {
        'total': total_loss,
        'cross_entropy': cross_entropy_loss,
        'entropy': entropy_loss,
    }
    return loss


def test(model, val_iter, vocab, reg_lambda, recon_lambda):
    model.eval()

    loss_checker = LossChecker(3)
    PAD_idx = vocab.word2idx['<PAD>']
    for b, batch in enumerate(val_iter, 1):
        _, feats, captions = parse_batch(batch)
        with torch.no_grad():
            output = model(feats)

        captions = captions.transpose(0, 1).contiguous()  # 变回原维度
        output = output.transpose(0, 1).contiguous()  # 输出维度变回原维度
        cross_entropy_loss = F.nll_loss(output[1:].view(-1, vocab.n_vocabs),
                                        captions[1:].contiguous().view(-1),
                                        ignore_index=PAD_idx)
        entropy_loss = losses.entropy_loss(output[1:], ignore_mask=(captions[1:] == PAD_idx))
        loss = cross_entropy_loss + reg_lambda * entropy_loss
        loss_checker.update(loss.item(), cross_entropy_loss.item(), entropy_loss.item())
    total_loss, cross_entropy_loss, entropy_loss = loss_checker.mean()
    loss = {
        'total': total_loss,
        'cross_entropy': cross_entropy_loss,
        'entropy': entropy_loss,
    }
    return loss


def get_predicted_captions(data_iter, model, vocab, beam_width=5, beam_alpha=0.):
    def build_onlyonce_iter(data_iter):
        onlyonce_dataset = {}
        for batch in iter(data_iter):
            vids, feats, _ = parse_batch(batch)
            for vid, feat in zip(vids, feats):
                if vid not in onlyonce_dataset:
                    onlyonce_dataset[vid] = feat
        onlyonce_iter = []
        vids = onlyonce_dataset.keys()
        feats = onlyonce_dataset.values()
        batch_size = 100
        while len(vids) > 0:
            onlyonce_iter.append(( list(vids)[:batch_size], torch.stack(list(feats)[:batch_size]) ))
            vids = list(vids)[batch_size:]
            feats = list(feats)[batch_size:]
        return onlyonce_iter

    model.eval()

    onlyonce_iter = build_onlyonce_iter(data_iter)

    vid2pred = {}
    EOS_idx = vocab.word2idx['<EOS>']
    for vids, feats in onlyonce_iter:
        with torch.no_grad():
            captions = model.module.cap_gen.describe(feats)
        captions = [ idxs_to_sentence(caption, vocab.idx2word, EOS_idx) for caption in captions ]
        vid2pred.update({ v: p for v, p in zip(vids, captions) })
    return vid2pred


# def get_groundtruth_captions(data_iter, vocab):
#     vid2GTs = {}
#     EOS_idx = vocab.word2idx['<EOS>']
#     for batch in iter(data_iter):
#         vids, _, captions = parse_batch(batch)
#         #captions = captions.transpose(0, 1)
#         for vid, caption in zip(vids, captions):
#             if vid not in vid2GTs:
#                 vid2GTs[vid] = []
#             caption = idxs_to_sentence(caption, vocab.idx2word, EOS_idx)
#             vid2GTs[vid].append(caption)
#     return vid2GTs

def get_groundtruth_captions_modify(data_iter,caption_path):
    captions=load_captions(caption_path)
    for batch in iter(data_iter):
        vids, _, _ = parse_batch(batch)
        for vid in vids:
            assert (vid in list(captions.keys()))
    return captions
def load_captions(caption_fpath):
    transform = transforms.Compose([

        Lowercase(),
        RemovePunctuation(),
        SplitWithWhiteSpace(),
        Truncate(TEST.max_caption_len),
        TrimExceptAscii(TEST.corpus),
    ])
    captions = {}
    if TEST.corpus=="MSVD":
        df = pd.read_csv(caption_fpath)
        df = df[df['Language'] == 'English']
        df = df[[ 'VideoID', 'Start', 'End', 'Description' ]]
        df = df[pd.notnull(df['Description'])]
        for video_id, start, end, caption in df.values:
            vid = "{}_{}_{}".format(video_id, start, end)
            caption = transform(caption)
            caption = ' '.join(caption)
            if vid not in captions:
                captions[vid] = []
            captions[vid].append(caption)
    if TEST.corpus=="MSR-VTT":
        with open(caption_fpath, 'r') as fin:
            data = json.load(fin)
        for vid, depth1 in data.items():
            for sid, caption in depth1.items():
                caption = transform(caption)
                caption = ' '.join(caption)
                if vid not in captions:
                    captions[vid] = []
                captions[vid].append(caption)
    return captions

def score(vid2pred, vid2GTs):
    assert set(vid2pred.keys()) == set(vid2GTs.keys())
    vid2idx = { v: i for i, v in enumerate(vid2pred.keys()) }
    refs = { vid2idx[vid]: GTs for vid, GTs in vid2GTs.items() }
    hypos = { vid2idx[vid]: [ pred ] for vid, pred in vid2pred.items() }

    for i in range(10):
        video_id = random.choice(list(vid2idx.keys()))

        print('\n', '%s:' % video_id, '\n ref: %s' % refs[vid2idx[video_id]][0],
              '\n pred: %s' % hypos[vid2idx[video_id]][0])
    print('\n')

    scores = calc_scores(refs, hypos)
    return scores


# refers: https://github.com/zhegan27/SCN_for_video_captioning/blob/master/SCN_evaluation.py
def calc_scores(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


def evaluate(data_iter, model, vocab,caption_path, beam_width=5, beam_alpha=0.):
    # vid2pred = get_predicted_captions(data_iter, model, vocab, beam_width=5, beam_alpha=0.)
    # vid2GTs = get_groundtruth_captions(data_iter, vocab)
    vid2pred = get_predicted_captions(data_iter, model, vocab, beam_width=beam_width, beam_alpha=beam_alpha)
    vid2GTs = get_groundtruth_captions_modify(data_iter, caption_path)
    scores = score(vid2pred, vid2GTs)
    return scores


# refers: https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def idxs_to_sentence(idxs, idx2word, EOS_idx):
    words = []
    # for idx in idxs[1:]:
    for idx in idxs:
        idx = idx.item()
        if idx == EOS_idx:
            break
        word = idx2word[idx]
        words.append(word)
    sentence = ' '.join(words)
    return sentence


def cls_to_dict(cls):
    properties = dir(cls)
    properties = [ p for p in properties if not p.startswith("__") ]
    d = {}
    for p in properties:
        v = getattr(cls, p)
        if inspect.isclass(v):
            v = cls_to_dict(v)
            v['was_class'] = True
        d[p] = v
    return d


# refers https://stackoverflow.com/questions/1305532/convert-nested-python-dict-to-object
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def dict_to_cls(d):
    cls = Struct(**d)
    properties = dir(cls)
    properties = [ p for p in properties if not p.startswith("__") ]
    for p in properties:
        v = getattr(cls, p)
        if isinstance(v, dict) and 'was_class' in v and v['was_class']:
            v = dict_to_cls(v)
        setattr(cls, p, v)
    return cls


def load_checkpoint(model, ckpt_fpath):
    checkpoint = torch.load(ckpt_fpath)
    model.module.cap_gen.load_state_dict(checkpoint['cap_gen'])
    return model


def save_checkpoint(e, model, ckpt_fpath, config):
    ckpt_dpath = os.path.dirname(ckpt_fpath)
    if not os.path.exists(ckpt_dpath):
        os.makedirs(ckpt_dpath)

    torch.save({
        'epoch': e,
        'cap_gen': model.module.cap_gen.state_dict(),
        'config': cls_to_dict(config),
    }, ckpt_fpath)


def save_result(vid2pred, vid2GTs, save_fpath):
    assert set(vid2pred.keys()) == set(vid2GTs.keys())

    save_dpath = os.path.dirname(save_fpath)
    if not os.path.exists(save_dpath):
        os.makedirs(save_dpath)

    vids = vid2pred.keys()
    with open(save_fpath, 'w') as fout:
        for vid in vids:
            GTs = ' / '.join(vid2GTs[vid])
            pred = vid2pred[vid]
            line = ', '.join([ str(vid), pred, GTs ])
            fout.write("{}\n".format(line))

