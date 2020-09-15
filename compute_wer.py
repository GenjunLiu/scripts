import os
import sys
import numpy as np

hypothesis_list = [
    "i studying like english very much", "i like studying english very much",
    "i studying english very very much", "i like studying very english much",
    "i like english very much", "like i i studying english very very much"
]

reference_list = [
    "i like studying english very much", "i like studying english very much",
    "i like studying english very much", "i like studying english very much",
    "i like studying english very much", "i like studying english very much"
]


def txt_score(hypos, refer):
  """

    compute the err number

    For example:

        refer = A B C
        hypos = X B V C

        after alignment

        refer = ~A B ^  C
        hypos = ~X B ^V C
        err number = 2

        where ~ denotes replacement error, ^ denote insertion error, * denotes deletion error.

    if set the special_word (not None), then the special word in reference can match any words in hypothesis.
    For example:
        refer = 'A <?> C'
        hypos = 'A B C D C'

        after aligment:

        refer =  A   <?> C
        hypos =  A B C D C

        where <?> matches 'B C D'. error number = 0

    Usage:
        ```
        refer = 'A <?> C'
        hypos = 'A B C D C'

        res = wer.wer(refer, hypos, '<?>')
        print('err={}'.format(res['err']))
        print('ins={ins} del={del} rep={rep}'.format(**res))
        print('refer = {}'.format(' '.join(res['refer'])))
        print('hypos = {}'.format(' '.join(res['hypos'])))
        ```

    Args:
        hypos: a string or a list of words
        refer: a string or a list of hypos
        special_word: this word in reference can match any words

    Returns:
        a result dict, including:
        res['word']: word number
        res['err']:  error number
        res['del']:  deletion number
        res['ins']:  insertion number
        res['rep']:  replacement number
        res['hypos']: a list of words, hypothesis after alignment
        res['refer']: a list of words, reference after alignment
    """

  res = {
      'word': 0,
      'err': 0,
      'none': 0,
      'del': 0,
      'ins': 0,
      'rep': 0,
      'hypos': [],
      'refer': []
  }

  refer_words = refer if isinstance(refer, list) else refer.split()
  hypos_words = hypos if isinstance(hypos, list) else hypos.split()

  hypos_words.insert(0, '<s>')
  hypos_words.append('</s>')
  refer_words.insert(0, '<s>')
  refer_words.append('</s>')

  hypos_len = len(hypos_words)
  refer_len = len(refer_words)

  if hypos_len == 0 or refer_len == 0:
    return res

  go_nexts = [[0, 1], [1, 1], [1, 0]]
  score_table = [([['none', 10000, [-1, -1], '', '']] * refer_len)
                 for hypos_cur in range(hypos_len)]
  # [error-type, note distance, best previous]
  score_table[0][0] = ['none', 0, [-1, -1], '', '']

  for hypos_cur in range(hypos_len - 1):
    for refer_cur in range(refer_len):

      for go_nxt in go_nexts:
        hypos_next = hypos_cur + go_nxt[0]
        refer_next = refer_cur + go_nxt[1]
        if hypos_next >= hypos_len or refer_next >= refer_len:
          continue

        next_score = score_table[hypos_cur][refer_cur][1]
        next_state = 'none'
        next_hypos = ''
        next_refer = ''

        if go_nxt == [0, 1]:
          next_state = 'del'
          next_score += 1
          next_hypos = '*' + ' ' * len(refer_words[refer_next])
          next_refer = '*' + refer_words[refer_next]

        elif go_nxt == [1, 0]:
          next_state = 'ins'
          next_score += 1
          next_hypos = '^' + hypos_words[hypos_next]
          next_refer = '^' + ' ' * len(hypos_words[hypos_next])

        else:
          next_hypos = hypos_words[hypos_next]
          next_refer = refer_words[refer_next]
          if hypos_words[hypos_next] != refer_words[refer_next]:
            next_state = 'rep'
            next_score += 1
            next_hypos = '~' + next_hypos
            next_refer = '~' + next_refer

        if next_score < score_table[hypos_next][refer_next][1]:
          score_table[hypos_next][refer_next] = [
              next_state, next_score, [hypos_cur, refer_cur], next_hypos,
              next_refer
          ]

  res['err'] = score_table[hypos_len - 1][refer_len - 1][1]
  res['word'] = refer_len - 2
  hypos_cur = hypos_len - 1
  refer_cur = refer_len - 1
  refer_fmt_words = []
  hypos_fmt_words = []
  while hypos_cur >= 0 and refer_cur >= 0:
    res[score_table[hypos_cur][refer_cur]
        [0]] += 1  # add the del/rep/ins error number
    hypos_fmt_words.append(score_table[hypos_cur][refer_cur][3])
    refer_fmt_words.append(score_table[hypos_cur][refer_cur][4])
    [hypos_cur, refer_cur] = score_table[hypos_cur][refer_cur][2]

  refer_fmt_words.reverse()
  hypos_fmt_words.reverse()

  # format the hypos and refer
  assert len(refer_fmt_words) == len(hypos_fmt_words)
  for hypos_cur in range(len(refer_fmt_words)):
    w = max(len(refer_fmt_words[hypos_cur]), len(hypos_fmt_words[hypos_cur]))
    fmt = '{:>%d}' % w
    refer_fmt_words[hypos_cur] = fmt.format(refer_fmt_words[hypos_cur])
    hypos_fmt_words[hypos_cur] = fmt.format(hypos_fmt_words[hypos_cur])

  res['refer'] = refer_fmt_words[1:-1]
  res['hypos'] = hypos_fmt_words[1:-1]

  return res


def compute_wer(hypothesis_list, reference_list):
  """
    input a set of (hypothesis, reference), compute the final wer

    Args:
        hypos_refer_iter: a iter return a tuple of (hypos , refer) or a tuple of (hypos, refer, key)
            the refer and hypos can be either word list or a sentence string
        special_word: special word
        output_stream: output stream

    Returns:
        (total_err, total_word, WER)
    """
  total_word = 0
  total_err = 0
  total_ins_err = 0
  total_del_err = 0
  total_sub_err = 0

  if len(hypothesis_list) != len(reference_list):
    print("Length mismatch, hypo: {}, ref: {}".format(len(hypothesis_list),
                                                      len(reference_list)))
    return

  for i in range(0, len(hypothesis_list)):
    res = txt_score(hypothesis_list[i], reference_list[i])
    total_err += res['err']
    total_ins_err += res['ins']
    total_del_err += res['del']
    total_sub_err += res['rep']
    total_word += res['word']

  print('total_err = {} | ins={} del={} sub={}'.format(total_err, total_ins_err,
                                                       total_del_err,
                                                       total_sub_err))
  print('total_word = {}'.format(total_word))
  print('wer = {:.6f}'.format(100.0 * total_err / total_word))


compute_wer(hypothesis_list, reference_list)
