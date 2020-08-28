import copy
from collections import defaultdict

# key: feature name
# value: wanted count
feature_list = {"feat_a": 2, "feat_b": 2, "feat_c": 2, "feat_d": 1, "feat_e": 2}

# sentence to features
sent_to_feat = {
    "sent_0": ["feat_a", "feat_b"],
    "sent_1": ["feat_d"],
    "sent_2": ["feat_c", "feat_d"],
    "sent_3": ["feat_a", "feat_d"],
    "sent_4": ["feat_a", "feat_b", "feat_c", "feat_d"],
    "sent_5": ["feat_b"],
    "sent_6": ["feat_a", "feat_b", "feat_e"],
    "sent_7": ["feat_a", "feat_b", "feat_e"],
    "sent_8": ["feat_a", "feat_c"],
    "sent_9": ["feat_b", "feat_c", "feat_e"],
    "sent_10": ["feat_b", "feat_e"],
    "sent_11": ["feat_a", "feat_d", "feat_e"],
    "sent_12": ["feat_a", "feat_d", "feat_e"],
}


def find_min_needed_sents(feat_sent_mat, want_feat_count, index):
  all_zero = True
  for count in want_feat_count:
    if count != 0:
      all_zero = False
  # all want feature count is ZERO, finish
  if all_zero:
    return True, []

  # fail to find
  if index >= len(feat_sent_mat[0]):
    return False, []

  # select current sentence
  new_count = copy.deepcopy(want_feat_count)
  for i in range(0, len(want_feat_count)):
    if not feat_sent_mat[i][index]:
      continue
    if new_count[i] > 0:
      new_count[i] -= 1
  success_1, sent_list_1 = find_min_needed_sents(feat_sent_mat, new_count,
                                                 index + 1)
  # select current sentence, append
  sent_list_1.append(index)

  # drop current sentence, do not modify `want_feat_count`
  success_2, sent_list_2 = find_min_needed_sents(feat_sent_mat, want_feat_count,
                                                 index + 1)
  if not success_2:
    return success_1, sent_list_1
  if not success_1:
    return False, []
  # sentence list 2 is better
  if len(sent_list_1) > len(sent_list_2):
    return True, sent_list_2
  return True, sent_list_1


def find_the_minimum_num_of_sentences(feature_list, sent_to_feat):
  index_to_feat_name = {}
  feat_name_to_index = {}
  want_feat_count = []
  total_feat_count = []
  index = 0
  for k, v in feature_list.items():
    index_to_feat_name[index] = k
    feat_name_to_index[k] = index
    want_feat_count.append(v)
    total_feat_count.append(0)
    index += 1

  feat_sent_mat = [None] * len(feature_list)
  for i in range(0, len(feature_list)):
    feat_sent_mat[i] = [False] * len(sent_to_feat)

  index_to_sent_name = {}
  sent_to_feat_index = []
  index = 0
  for k, v in sent_to_feat.items():
    index_to_sent_name[index] = k
    sent_to_feat_index.append([])
    for feat in v:
      if not feat in feat_name_to_index:
        continue
      feat_index = feat_name_to_index[feat]
      sent_to_feat_index[index].append(feat_index)
      feat_sent_mat[feat_index][index] = True
      total_feat_count[feat_index] += 1
    index += 1

  # debug message
  # print(index_to_feat_name)
  # print(index_to_sent_name)
  # print(feat_sent_mat)
  rest_feat_count = [0] * len(total_feat_count)
  for i in range(0, len(total_feat_count)):
    if total_feat_count[i] < want_feat_count[i]:
      rest_feat_count[i] = want_feat_count[i] - total_feat_count[i]
      want_feat_count[i] = total_feat_count[i]
  success, min_sent_list = find_min_needed_sents(feat_sent_mat, want_feat_count,
                                                 0)
  min_sent_name_list = []
  for index in min_sent_list:
    min_sent_name_list.append(index_to_sent_name[index])

  rest_feat_list = {}
  for i in range(0, len(rest_feat_count)):
    if rest_feat_count[i] == 0:
      continue
    rest_feat_list[index_to_feat_name[i]] = rest_feat_count[i]
  if len(rest_feat_list) != 0:
    success = False
  return success, rest_feat_list, min_sent_name_list


# run
success, rest_feat_list, min_sent_list = find_the_minimum_num_of_sentences(
    feature_list, sent_to_feat)

if success:
  print("selected sentence list(success):")
  print(min_sent_list)
else:
  print("selected sentence list(fail):")
  print(min_sent_list)
  print("rest feat:")
  print(rest_feat_list)
