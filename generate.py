#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import torch
import json
import numpy as np
from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from sacrebleu.metrics import BLEU
from torch._C import device


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu
    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()
    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,    
        max_sentences=args.max_sentences,   
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    
    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True
    
#   sent_number = task.dataset(args.gen_subset).src_sizes.size
    cross_attention_weights = None
    allResult_socre = torch.tensor(0.0).cuda()
    allResult_num = torch.tensor(0.0).cuda()
    rws=[]

    bleu_need = BLEU()
#    res_BLEU = []
#    res_sentences = []
#    sentences_id = []
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue
#            sentences_id.append(sample['id'])

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]
            gen_timer.start()
            hypos, g,src_lens,dict_all = task.inference_step(generator, models, sample, prefix_tokens, cross_attention_weights,waitK=None)  
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)
            tmp_rws = [d201(g[i],src_lens[i]) for i in range(len(g))]
            rws.extend(tmp_rws)

            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None

                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                if not args.quiet:
                    if src_dict is not None:
                        print('S-{}\t{}'.format(sample_id, src_str))
                    if has_target:
                        print('T-{}\t{}'.format(sample_id, target_str))

                # Process top predictions
                for j, hypo in enumerate(hypos[i][:args.nbest]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )

                    if not args.quiet:
                        print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                        print('P-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                hypo['positional_scores'].tolist(),
                            ))
                        ))
                        print(tmp_rws[dict_all[i]])
                        if args.print_alignment:
                            print('A-{}\t{}'.format(
                                sample_id,
                                ' '.join(map(lambda x: str(utils.item(x)), alignment))
                            ))

                    # Score only the top hypothesis
                    if has_target and j == 0:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                        if hasattr(scorer, 'add_string'):
                            scorer.add_string(target_str, hypo_str)
                        else:
                            scorer.add(target_tokens, hypo_tokens)                        
                    
#                        tmp_bleu_str = bleu_need.corpus_score([hypo_str], [[target_str]])
#                        tmp_bleu_float = float(tmp_bleu_str.score)
#                        res_BLEU.append(tmp_bleu_float)
#                        res_sentences.append((src_str,hypo_str))
                    
            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']

    cw,ap,al,dal=compute_delay(rws, is_weight_ave=True)

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))
    print('AL: {}; AP: {}; CW: {}; DAL:{}'.format(al, ap, cw, dal))
    return scorer

def d201(d,src):
    #print("+++",d)
    s='0 '*int(d[0]+1)+'1 '
    for i in range(1,len(d)):
        s=s+'0 '*int((min(d[i],src-1)-min(d[i-1],src-1)))+'1 '
    if src>d[-1]+1:
        s=s+'0 '*(src-d[-1]-1)
    return s


def d201_leftpad(d,src,bsz_len):
    #print("+++",d)
    if src<bsz_len: d=[max(1,a-(bsz_len-src)) for a in d]
    s='0 '*int(d[0]+1)+'1 '
    for i in range(1,len(d)):
        s=s+'0 '*int((min(d[i],src-1)-min(d[i-1],src-1)))+'1 '
    if src>d[-1]+1:
        s=s+'0 '*(src-d[-1]-1)
    return s


def generate_rw(src_len,tgt_len,k):
    rws=[]
    gs=[]
    for i in range(0, len(src_len)):
        g=''
        for j in range(0,tgt_len[i]-1):
            g+=str(min(k+j,src_len[i]-1))+' '
        if src_len[i] <= k:
            rw = '0 ' * src_len[i] + '1 ' * tgt_len[i]
        else:
            if tgt_len[i] + k > src_len[i]:
                rw = '0 ' * k + '1 0 ' * (src_len[i] - k) + '1 ' * (tgt_len[i] + k - src_len[i])
            else:
                rw = '0 ' * k + '1 0 ' * (tgt_len[i] ) + '0 ' * (src_len[i] - tgt_len[i] - k)
        rws.append(rw)
        gs.append(g)
    #print(rws)
    return rws,gs

def compute_delay(rw, is_weight_ave=False):
    CWs,ALs,APs,DALs, Lsrc =[],[], [], [],[]
    for line in rw:
        line = line.strip()
        al_ans = RW2AL(line,add_eos=False)
        dal_ans = RW2DAL(line,add_eos=False)
        ap_ans=RW2AP(line,add_eos=False)
        cw_ans = RW2CW(line,add_eos=False)
        if al_ans is not None:
            ALs.append(al_ans)
            DALs.append(dal_ans)
            APs.append(ap_ans)
            CWs.append(cw_ans)
            Lsrc.append(line.count('0'))

    CW = np.average(CWs) if is_weight_ave else np.average(CWs, weights=Lsrc)
    AL = np.average(ALs) if is_weight_ave else np.average(ALs, weights=Lsrc)
    DAL = np.average(DALs) if is_weight_ave else np.average(DALs, weights=Lsrc)
    AP = np.average(APs) if is_weight_ave else np.average(APs, weights=Lsrc)
    return CW,AP,AL,DAL


def RW2CW(s, add_eos=False):
    trantab = str.maketrans('RrWw', '0011')
    if isinstance(s, str):
        s = s.translate(trantab).replace(' ','').replace(',','')
        if add_eos: # to add eos token for both src and tgt if you did not do it during RW generating
            idx = s.rfind('0')
            s = s[:idx+1]+'0'+s[idx+1:]+'1'  # remove last 0/1(<eos>) to keep actuall setence length
            # s = (s[:idx]+s[idx+1:])[:-1]  # remove last 0/1(<eos>) to keep actuall setence length
    else: return None
    x, y = s.count('0'), s.count('1')
    if x == 0 or y == 0: return 0
    c=s.count('01')

    if c==0:
        return 0
    else:
        return x/c

# s is RW sequence, in format of '0 0 0 1 1 0 1 0 1', or 'R R R W W R W R W', flexible on blank/comma
def RW2AP(s, add_eos=False):
    trantab = str.maketrans('RrWw', '0011')
    if isinstance(s, str):
        s = s.translate(trantab).replace(' ','').replace(',','')
        if add_eos: # to add eos token for both src and tgt if you did not do it during RW generating
            idx = s.rfind('0')
            s = s[:idx+1]+'0'+s[idx+1:]+'1'  # remove last 0/1(<eos>) to keep actuall setence length
            # s = (s[:idx]+s[idx+1:])[:-1]  # remove last 0/1(<eos>) to keep actuall setence length
    else: return None
    x, y = s.count('0'), s.count('1')
    if x == 0 or y == 0: return 0

    count = 0
    curr = []
    for i in s:
        if i == '0': count += 1
        else: curr.append(count)
    return sum(curr) / x / y


# s is RW sequence, in format of '0 0 0 1 1 0 1 0 1', or 'R R R W W R W R W', flexible on blank/comma
def RW2AL(s, add_eos=False):
    trantab = str.maketrans('RrWw', '0011')
    if isinstance(s, str):
        s = s.translate(trantab).replace(' ','').replace(',','')
        if add_eos: # to add eos token for both src and tgt if you did not do it during RW generating
            idx = s.rfind('0')
            s = s[:idx+1]+'0'+s[idx+1:]+'1'  # remove last 0/1(<eos>) to keep actuall setence length
            # s = (s[:idx]+s[idx+1:])[:-1]  # remove last 0/1(<eos>) to keep actuall setence length
    else: return None
    x, y = s.count('0'), s.count('1')
    if x == 0 or y == 0: return 0

    count = 0
    rate = y/x
    curr = []
    for i in s:
        if i == '0': count += 1
        else: curr.append(count)
        if i == '1' and count == x: break
    y1 = len(curr)
    diag = [(t-1)/rate for t in range(1, y1+1)]
    return sum(l1-l2 for l1,l2 in zip(curr,diag)) / y1

def RW2DAL(s, add_eos=False):
    trantab = str.maketrans('RrWw', '0011')
    if isinstance(s, str):
        s = s.translate(trantab).replace(' ','').replace(',','')
        if add_eos: # to add eos token for both src and tgt if you did not do it during RW generating
            idx = s.rfind('0')
            s = s[:idx+1]+'0'+s[idx+1:]+'1'  # remove last 0/1(<eos>) to keep actuall setence length
            # s = (s[:idx]+s[idx+1:])[:-1]  # remove last 0/1(<eos>) to keep actuall setence length
    else: return None
    x, y = s.count('0'), s.count('1')
    if x == 0 or y == 0: return 0

    count = 0
    rate = y/x
    curr = []
    curr1=[]
    for i in s:
        if i == '0': count += 1
        else: curr.append(count)
    curr1.append(curr[0])
    for i in range(1, y):
        curr1.append(max(curr[i],curr1[i-1]+1/rate))

    diag = [(t-1)/rate for t in range(1, y+1)]
    return sum(l1-l2 for l1,l2 in zip(curr1,diag)) / y



def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()