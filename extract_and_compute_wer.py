# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import json
from collections import defaultdict
from compute_wer import compute_wer


def parse_args():
  parser = argparse.ArgumentParser(description="test openscorer")
  parser.add_argument("--input_json",
                      type=str,
                      default="result.json",
                      help="input json file")
  parser.add_argument("--output", type=str, default="result.txt", help="output")
  args = parser.parse_args()
  return args


def extract_result(args):
  f = open(args.output, "w")

  header = [
      "object_id", "close_time", "1_审核人", "1_have_noise", "1_is_discard",
      "1_phone_text", "1_text", "2_审核人", "2_have_noise", "2_is_discard",
      "2_phone_text", "2_text", "audio_url"
  ]
  f.write("{}\n".format("\t".join(header)))

  ###################################
  # dead code
  # falsify data
  line_no = 0
  ###################################
  verifier_to_text = defaultdict(list)
  verifier_to_ref_text = defaultdict(list)
  for line in open(args.input_json).readlines():
    js_data = json.loads(line.strip())
    fields = []
    close_time = js_data["closed_time"]
    object_id = js_data["object_data"]["object_id"]
    audio_url = js_data["object_data"]["audio_url"]
    verifies = js_data["verifies"]
    fields.append(object_id)
    fields.append(close_time)
    ###################################
    # dead code
    js_data["verify_count"] = 2
    verifies.append(verifies[0])
    line_no += 1
    ###################################
    verify_count = js_data["verify_count"]
    if verify_count != 2:
      continue

    if len(verifies) != 2:
      print("Error: Actually verify_count is {}".format(len(verifies)))
      continue

    verifier_name = []
    phone_texts = []
    for v in verifies:
      person = v["verifier"]
      ###################################
      # dead code
      if line_no % 2 == 0:
        person = "luojiawen.1017"
      ###################################
      fields.append(person)
      verifier_name.append(person)
      main_form = v["verify_data"]["mainForm"]
      data_dict = {}
      for v in main_form:
        if v["name"] == "is_discard":
          data_dict["is_discard"] = v["value"]
        elif v["name"] == "have_noise":
          data_dict["have_noise"] = v["value"]
        elif v["name"] == "phone_text":
          data_dict["phone_text"] = v["value"]
        elif v["name"] == "text":
          data_dict["text"] = v["value"]

      fields.append(data_dict["have_noise"])
      fields.append(data_dict["is_discard"])
      fields.append(data_dict["phone_text"])
      fields.append(data_dict["text"])
      phone_texts.append(data_dict["phone_text"])
    fields.append(audio_url)

    ###################################
    # dead code
    temp_text = phone_texts[1]
    phone_list = temp_text.split(" ")
    if line_no % 3 == 0 and len(phone_list) >= 2:
      phone_list.append(phone_list[len(phone_list) - 2])
    phone_texts[1] = " ".join(phone_list)
    ###################################
    verifier_to_text[verifier_name[0]].append(phone_texts[0])
    verifier_to_ref_text[verifier_name[0]].append(phone_texts[1])
    # skip
    if phone_texts[0] == phone_texts[1]:
      continue
    f.write("{}\n".format("\t".join(fields)))
  f.close()

  for k in verifier_to_text.keys():
    print("==== {} ====".format(k))
    compute_wer(verifier_to_text[k], verifier_to_ref_text[k])
    print("")


if __name__ == "__main__":
  args = parse_args()
  extract_result(args)
