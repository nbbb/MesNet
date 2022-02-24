import random

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .allennlp_beamsearch import BeamSearch

class CaptionGenerator(nn.Module):
    def __init__(self, decoder, max_caption_len, vocab,embedding_size,vocab_pretrain,beam_size=1):
        super(CaptionGenerator, self).__init__()
        self.decoder = decoder
        self.max_caption_len = max_caption_len
        self.vocab = vocab
        self.beam_size=beam_size
        self.output_size=vocab.n_vocabs
        self.embedding_size=embedding_size
        self.beam_search = BeamSearch(self.vocab.word2idx['<SOS>'], self.max_caption_len, self.beam_size, per_node_beam_size=self.beam_size)
        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        if vocab_pretrain:
            glove_pretrain = np.load(vocab_pretrain)
            self.embedding.weight.data.copy_(torch.from_numpy(glove_pretrain))
    def get_rnn_init_hidden(self, batch_size, num_layers, num_directions, hidden_size):
        hidden = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
        hidden = hidden.cuda()
        return hidden

    def forward_decoder(self, batch_size, vocab_size, hidden, feats, captions, teacher_forcing_ratio):
        outputs = Variable(torch.zeros(self.max_caption_len + 2, batch_size, vocab_size)).cuda()
        output = Variable(torch.cuda.LongTensor(1, batch_size).fill_(self.vocab.word2idx['<SOS>']))
        embedded=self.embedding(output.view(1, -1))
        for t in range(1, self.max_caption_len + 2):
            output, hidden, attn_weights = self.decoder(embedded, hidden, feats)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(captions.data[t] if is_teacher else top1).cuda()
            embedded=self.embedding(output.view(1, -1))
        return outputs




    def forward(self, feats, captions=None, teacher_forcing_ratio=0.):
        batch_size = feats.size(0)
        if captions is not None:
            captions = captions.transpose(0, 1).contiguous()
        vocab_size = self.decoder.output_size

        hidden = self.get_rnn_init_hidden(batch_size, self.decoder.num_layers, self.decoder.num_directions,
                                          self.decoder.hidden_size)

        outputs = self.forward_decoder(batch_size, vocab_size, hidden, feats, captions,
                                                        teacher_forcing_ratio)
        outputs = outputs.transpose(0, 1).contiguous()
        return outputs

    def describe(self, feats):
        batch_size = feats.size(0)
        hidden = self.get_rnn_init_hidden(batch_size, self.decoder.num_layers, self.decoder.num_directions,
                                          self.decoder.hidden_size)
        start_state={'hidden':hidden.permute(1,0,2).contiguous(),'feats':feats}
        start_id = Variable(torch.cuda.LongTensor(batch_size).fill_(self.vocab.word2idx['<SOS>']))
        captions = self.beam_create_caption(start_id, start_state,batch_size)
        return captions

    def beam_create_caption(self, start_id, start_state,batch_size):
        outputs = []
        predictions, log_prob = self.beam_search.search(start_id, start_state, self.beam_step)
        max_prob, max_index = torch.topk(log_prob, 1)
        max_index = max_index.squeeze(1)
        for i in range(batch_size):
            outputs.append(predictions[i, max_index[i], :])
        outputs = torch.stack(outputs)
        if outputs.size(1) < self.max_caption_len:
            pad_outs = Variable(torch.zeros(outputs.size(0), self.max_caption_len - outputs.size(1), dtype=int)).cuda()
            outputs = torch.cat((outputs, pad_outs), dim=1)  # 向量拼接
        return outputs


    def beam_step(self, last_predictions, current_state):
        group_size = last_predictions.size(0)
        batch_size = current_state['feats'].size(0)
        log_probs = []
        new_state = {}
        num = int(group_size / batch_size)
        for k, state in current_state.items():
            if isinstance(state, list):
                state = torch.stack(state, dim=1)
            _, *last_dims = state.size()
            current_state[k] = state.reshape(batch_size, num, *last_dims)
            new_state[k] = []
        for i in range(num):
            # read current state
            hidden = current_state['hidden'][:, i, :].permute(1,0,2).contiguous()
            feats = current_state['feats'][:, i, :]
            # decoding stage
            word_id = last_predictions.reshape(batch_size, -1)[:, i]
            word = self.embedding(word_id.view(1, -1))
            output, hidden, attn_weights  = self.decoder(word, hidden, feats)

            # store log probabilities

            log_probs.append(output)

            # update new state
            new_state['hidden'].append(hidden.permute(1,0,2).contiguous())
            new_state['feats'].append(feats)

        # transform log probabilities
        # from list to tensor(batch_size*beam_size, vocab_size)
        log_probs = torch.stack(log_probs, dim=0).permute(1, 0, 2).reshape(group_size, -1)  # group_size*vocab_size

        # transform new state
        # from list to tensor(batch_size*beam_size, *)
        for k, state in new_state.items():
            new_state[k] = torch.stack(state, dim=0)  # (beam_size, batch_size, *)
            _, _, *last_dims = new_state[k].size()
            dim_size = len(new_state[k].size())
            dim_size = range(2, dim_size)
            new_state[k] = new_state[k].permute(1, 0, *dim_size)  # (batch_size, beam_size, *)
            new_state[k] = new_state[k].reshape(group_size, *last_dims)  # (batch_size*beam_size, *)
        return (log_probs, new_state)


