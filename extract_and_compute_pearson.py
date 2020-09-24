# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import json


# usage:
#python extract_and_compute_pearson.py --json_dir=json --output=temp_dir
def parse_args():
  parser = argparse.ArgumentParser(description="test openscorer")
  parser.add_argument("--json_dir", type=str, default="json", help="json dir")
  parser.add_argument("--output_dir", type=str, default="output", help="output")
  args = parser.parse_args()
  return args


def extract_result(filename):
  all_data = {}
  for line in open(filename).readlines():
    js_data = json.loads(line.strip())
    close_time = js_data["closed_time"]
    object_id = js_data["object_data"]["object_id"]
    wav_url = js_data["object_data"]["wav_url"]
    text = js_data["object_data"]["text"]
    verifies = js_data["verifies"]
    single_data = {}
    single_data["object_id"] = object_id
    single_data["close_time"] = close_time
    single_data["wav_url"] = wav_url
    single_data["text"] = text
    f = filter(None, text.split(" "))
    words = list(f)
    necessary_keys = {}
    for i in range(0, len(words)):
      name = "word_" + str(i + 1)
      necessary_keys[name] = words[i]
    necessary_keys["should_discard"] = "should_discard"
    necessary_keys["sentence_accuracy"] = "sentence_accuracy"
    necessary_keys["sentence_fluency"] = "sentence_fluency"

    single_data["verifiers"] = []
    for v in verifies:
      person = v["verifier"]
      main_form = v["verify_data"]["mainForm"]
      data_dict = {}
      data_dict["verifier"] = person
      for kv in main_form:
        key = kv["name"]
        if not key in necessary_keys:
          continue
        data_dict[necessary_keys[key]] = kv["value"]
      single_data["verifiers"].append(data_dict)

    first_verifier = verifies[0]["verifier"]
    if not first_verifier in all_data:
      all_data[first_verifier] = []
    all_data[first_verifier].append(single_data)
  return all_data


def convert_result(result):
  verifier_to_data = {}
  for k, v in result.items():
    verifier_to_data[k] = {}
    sentence_fluency = {}
    sentence_accuracy = {}
    sentence_words = {}
    for data in v:
      v_0 = data["verifiers"][0]
      v_1 = {}
      verified = (len(data["verifiers"]) == 2)
      if verified:
        v_1 = data["verifiers"][1]
      if v_0["should_discard"] == "1" or (verified and
                                          v_1["should_discard"] == "1"):
        continue
      object_id = data["object_id"]
      text = data["text"]
      f = filter(None, text.split(" "))
      words = list(f)
      word_to_index = {}
      word_result_0 = []
      word_result_1 = []
      for w in words:
        if not w in v_0:
          print("Error")
          continue
        if verified and (not w in v_1):
          print("Error")
          continue
        word_result_0.append(v_0[w])
        if verified:
          word_result_1.append(v_1[w])
      sentence_words[object_id] = [word_result_0]
      sentence_fluency[object_id] = [v_0["sentence_fluency"]]
      sentence_accuracy[object_id] = [v_0["sentence_accuracy"]]
      if verified:
        sentence_words[object_id].append(word_result_1)
        sentence_fluency[object_id].append(v_1["sentence_fluency"])
        sentence_accuracy[object_id].append(v_1["sentence_accuracy"])
    verifier_to_data[k]["sentence_words"] = sentence_words
    verifier_to_data[k]["sentence_fluency"] = sentence_fluency
    verifier_to_data[k]["sentence_accuracy"] = sentence_accuracy
  return verifier_to_data


def split_verifier(data):
  data_0 = []
  data_1 = []
  for k, v in data.items():
    if len(v) == 2:
      if v[0] is None:
        continue
      if v[1] is None:
        continue
      if isinstance(v[0], list):
        for i in range(0, len(v[0])):
          # skip None and invalid data
          if v[0][i] is None or v[0][i] == "invalid":
            continue
          if v[1][i] is None or v[1][i] == "invalid":
            continue
          data_0.append(v[0][i])
          data_1.append(v[1][i])
      else:
        data_0.append(v[0])
        data_1.append(v[1])
  return data_0, data_1


def get_object_data(data_1, data_2):
  data_vec_1 = []
  data_vec_2 = []
  for k, v in data_1.items():
    if not k in data_2:
      continue
    if data_1[k][0] is None:
      continue
    if data_2[k][0] is None:
      continue
    want_index_1 = 0
    want_index_2 = 0
    # use verified data
    if len(data_1[k]) == 2:
      want_index_1 = 1
    if len(data_2[k]) == 2:
      want_index_2 = 1
    if isinstance(data_1[k][want_index_1], list):
      for i in range(0, len(data_1[k][want_index_1])):
        # skip None and invalid data
        if data_1[k][want_index_1][i] is None or data_1[k][want_index_1][
            i] == "invalid":
          continue
        if data_2[k][want_index_2][i] is None or data_2[k][want_index_2][
            i] == "invalid":
          continue
        data_vec_1.append(data_1[k][want_index_1][i])
        data_vec_2.append(data_2[k][want_index_2][i])
    else:
      data_vec_1.append(data_1[k][want_index_1])
      data_vec_2.append(data_2[k][want_index_2])
  return data_vec_1, data_vec_2


def compute_pearson_and_accuracy(preds, labels, fp):
  if len(preds) == 0:
    fp.write("data count: {}\n".format(len(preds)))
    fp.write("pearson coef: {}\n".format(0))
    fp.write("accuracy: {}\n".format(0))
    return

  preds_int = [int(x) for x in preds]
  labels_int = [int(x) for x in labels]
  pearson_coef = np.corrcoef(preds_int, labels_int)
  fp.write("data count: {}\n".format(len(preds_int)))
  fp.write("pearson coef: {}\n".format(pearson_coef[0][1]))

  positive_count = 0
  for i in range(0, len(preds_int)):
    if preds_int[i] == labels_int[i]:
      positive_count += 1
  fp.write("accuracy: {}\n".format(float(positive_count) / len(preds_int)))


def compute_pearson(args):
  result = {}
  for f in os.listdir(args.json_dir):
    if not f.endswith(".json"):
      continue
    all_data = extract_result(os.path.join(args.json_dir, f))
    for k, v in all_data.items():
      if not k in result:
        result[k] = []
      result[k].extend(v)

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  # # debug
  # for k, v in result.items():
  #   with open(os.path.join(args.output_dir, k + '.json'), 'w') as fp:
  #     json.dump(v, fp)

  # sample:
  # {
  #   "text": "I\u2019m going to be in a bike camp.",
  #   "close_time": "2020-09-17 15:36:09",
  #   "wav_url": "http://tosv.byted.org/obj/speech-tts-general/ev/ev125.mp3",
  #   "verifiers": [
  #     {
  #       "a": "0",
  #       "be": "0",
  #       "camp.": "0",
  #       "to": "2",
  #       "sentence_accuracy": "1",
  #       "sentence_fluency": "2",
  #       "bike": "0",
  #       "I\u2019m": "1",
  #       "verifier": "lijinyue",
  #       "going": "2",
  #       "should_discard": "0",
  #       "in": "2"
  #     }
  #   ],
  #   "object_id": "ev125"
  # },
  object_to_data = {}
  for k, v in result.items():
    verifier = k
    for value in v:
      object_id = value["object_id"]
      if not object_id in object_to_data:
        object_to_data[object_id] = {}
        object_to_data[object_id]["wav_url"] = value["wav_url"]
        object_to_data[object_id]["text"] = value["text"]
        object_to_data[object_id]["verifiers"] = []
      object_to_data[object_id]["verifiers"].append(value["verifiers"][0])

  # debug
  # with open(os.path.join(args.output_dir, 'object_to_data.json'), 'w') as fp:
  #   json.dump(object_to_data, fp)

  fp = open(os.path.join(args.output_dir, 'verifier_data.tsv'), 'w')
  fp_bad = open(os.path.join(args.output_dir, 'verifier_data_bad.tsv'), 'w')
  fields = [
      "object_id", "wav_url", "text", "verifier_1", "should_discard_1",
      "sent_acc_1", "sent_flu_1", "word_score_1", "verifier_2",
      "should_discard_2", "sent_acc_2", "sent_flu_2", "word_score_2",
      "verifier_3", "should_discard_3", "sent_acc_3", "sent_flu_3",
      "word_score_3", "..."
  ]
  fp.write("{}\n".format('\t'.join(fields)))
  fp_bad.write("{}\n".format('\t'.join(fields)))
  for k, v in object_to_data.items():
    fields = [k, v["wav_url"], v["text"]]

    f = filter(None, v["text"].split(" "))
    words = list(f)

    is_bad_data = False
    for verifier in v["verifiers"]:
      fields.append(verifier["verifier"])
      fields.append(verifier["should_discard"])
      if verifier["sentence_accuracy"] is None:
        fields.append("None")
      else:
        fields.append(verifier["sentence_accuracy"])
      if verifier["sentence_fluency"] is None:
        fields.append("None")
      else:
        fields.append(verifier["sentence_fluency"])
      words_score = {}
      for w in words:
        words_score[w] = verifier[w]
        # bad data an should not discard
        if verifier[w] is None and verifier["should_discard"] == "0":
          is_bad_data = True
      fields.append(json.dumps(words_score))

    fp.write("{}\n".format('\t'.join(fields)))
    if is_bad_data:
      fp_bad.write("{}\n".format('\t'.join(fields)))
  fp.close()
  fp_bad.close()

  fp_result = open(os.path.join(args.output_dir, 'result.txt'), 'w')
  verifier_to_data = convert_result(result)
  for k, v in verifier_to_data.items():
    # verifier1 vs verifier2
    fp_result.write("====== verifier: {} ======\n".format(k))
    sent_acc_1, sent_acc_2 = split_verifier(v["sentence_accuracy"])
    fp_result.write("\n**** sentence_accuracy ****\n")
    compute_pearson_and_accuracy(sent_acc_1, sent_acc_2, fp_result)

    sent_fluency_1, sent_fluency_2 = split_verifier(v["sentence_fluency"])
    fp_result.write("\n**** sentence_fluency ****\n")
    compute_pearson_and_accuracy(sent_fluency_1, sent_fluency_2, fp_result)

    sent_words_1, sent_words_2 = split_verifier(v["sentence_words"])
    fp_result.write("\n**** sentence_words ****\n")
    compute_pearson_and_accuracy(sent_words_1, sent_words_2, fp_result)

    fp_result.write("\n")

    # debug
    # # write to file
    # with open(os.path.join(args.output_dir, k + '_convert.json'), 'w') as fp:
    #   json.dump(v, fp)

  data_keys = list(verifier_to_data.keys())
  for i in range(0, len(data_keys)):
    for j in range(i, len(data_keys)):
      if i == j:
        continue
      fp_result.write("==== {} vs. {} ====".format(data_keys[i], data_keys[j]))
      sent_acc_1, sent_acc_2 = get_object_data(
          verifier_to_data[data_keys[i]]["sentence_accuracy"],
          verifier_to_data[data_keys[j]]["sentence_accuracy"])
      fp_result.write("\n**** sentence_accuracy ****\n")
      compute_pearson_and_accuracy(sent_acc_1, sent_acc_2, fp_result)

      sent_fluency_1, sent_fluency_2 = get_object_data(
          verifier_to_data[data_keys[i]]["sentence_fluency"],
          verifier_to_data[data_keys[j]]["sentence_fluency"])
      fp_result.write("\n**** sentence_fluency ****\n")
      compute_pearson_and_accuracy(sent_fluency_1, sent_fluency_2, fp_result)

      sent_words_1, sent_words_2 = get_object_data(
          verifier_to_data[data_keys[i]]["sentence_words"],
          verifier_to_data[data_keys[j]]["sentence_words"])
      fp_result.write("\n**** sentence_words ****\n")
      compute_pearson_and_accuracy(sent_words_1, sent_words_2, fp_result)

      fp_result.write("\n")


if __name__ == "__main__":
  args = parse_args()
  compute_pearson(args)
