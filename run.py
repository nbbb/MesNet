from __future__ import print_function
import os
from torch import nn
import torch
import os
from utils import dict_to_cls, get_predicted_captions, save_result, score,load_checkpoint
from configs.run import RunConfig as C
from loader.MSVD import MSVD
from loader.MSRVTT import MSRVTT
from models.decoder_indrnn import IndRNN_Decoder
from models.caption_generator import CaptionGenerator
class Net(nn.Module):
    def __init__(self, C, vocab):
        super(Net, self).__init__()
        self.TIME_STEPS= C.loader.max_caption_len+2
        self.RECURRENT_MAX = pow(2, 1 / self.TIME_STEPS)
        self.RECURRENT_MIN = pow(1 / 2, 1 / self.TIME_STEPS)

        recurrent_inits = []
        for _ in range(C.decoder.rnn_num_layers - 1):
            recurrent_inits.append(
                lambda w: nn.init.uniform_(w, 0, self.RECURRENT_MAX)
            )
        recurrent_inits.append(lambda w: nn.init.uniform_(
            w, self.RECURRENT_MIN, self.RECURRENT_MAX))
        decoder = IndRNN_Decoder(
            num_layers=C.decoder.rnn_num_layers,
            num_directions=C.decoder.rnn_num_directions,
            feat_size=C.feat.size,
            feat_len=C.loader.frame_sample_len,
            embedding_size=C.vocab.embedding_size,
            hidden_size=C.decoder.rnn_hidden_size,
            attn_size=C.decoder.rnn_attn_size,
            output_size=vocab.n_vocabs,
            rnn_dropout=C.decoder.rnn_dropout,
            recurrent_inits=recurrent_inits,
            hidden_max_abs=self.RECURRENT_MAX,
            batch_norm=C.decoder.batch_norm,
            gradient_clip=5
            )

        self.cap_gen = CaptionGenerator(decoder, C.loader.max_caption_len, vocab,C.vocab.embedding_size,C.vocab.pre_trained_glove_fpath,C.beam_size)
    def forward(self,feats, captions=None, teacher_forcing_ratio=0.):
        outputs=self.cap_gen(feats,captions,teacher_forcing_ratio)
        return outputs

def run(ckpt_fpath):
    checkpoint = torch.load(ckpt_fpath)

    """ Load Config """
    config = dict_to_cls(checkpoint['config'])
    """ Build Data Loader """
    if config.corpus == "MSVD":
        corpus = MSVD(config)
    elif config.corpus == "MSR-VTT":
        corpus = MSRVTT(config)
    train_iter, val_iter, test_iter,vocab = \
        corpus.train_data_loader, corpus.val_data_loader, corpus.test_data_loader,corpus.vocab
    print('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
        vocab.n_vocabs, vocab.n_vocabs_untrimmed, vocab.n_words, vocab.n_words_untrimmed, config.loader.min_count))

    """ Build Models """
    model= Net(config,vocab)
    model.cuda()
    model = nn.DataParallel(model)
    load_checkpoint(model, ckpt_fpath)
    """ Test Set """
    test_vid2pred = get_predicted_captions(test_iter, model,model.module.cap_gen.vocab)
    test_vid2GTs = test_iter.captions
    test_scores = score(test_vid2pred, test_vid2GTs)
    print("[TEST] {}".format(test_scores))

    test_save_fpath = os.path.join(C.result_dpath, "{}_{}.csv".format(config.corpus, 'test'))
    save_result(test_vid2pred, test_vid2GTs, test_save_fpath)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    run(C.ckpt_fpath)

