from collections import defaultdict
import numpy as np
from typing import Dict, Set, Tuple
from copy import deepcopy

def create_indicies_pairs(row: int, col: int, length: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    begin_first = row-length+1
    begin_second = col-length+1
    end_first = row
    end_second = col
    return (int(begin_first), int(end_first)), (int(begin_second), int(end_second))

def overlapping(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2
    return not (end1 < start2 or start1 > end2)

def store_found_repeat(protein1, result_dict1, row1, col1, length1):
  first, second = create_indicies_pairs(row1, col1, length1)
  key = protein1[first[0]:first[1]+1]
  result_dict1[key].add(first)
  result_dict1[key].add(second)

def FAIR_algorithm(protein: str, min: int, allow_overlapping=False) -> Dict[str, Set[Tuple[int, int]]]:
  #Based on https://core.ac.uk/download/pdf/291560955.pdf
  assert type(protein) == str, "protein must be a string"
  res =  defaultdict(set)

  y = len(protein)
  previous = np.zeros(shape=y, dtype=np.int32)
  current = np.zeros(shape=y, dtype=np.int32)

  for i in range(y):
    for j in range(i + 1, y):
      end_repeat = False
      if protein[i] == protein[j]:
        first,second = create_indicies_pairs(i, j, previous[j-1]+1)
        if overlapping(first, second) and not allow_overlapping:
          current[j] = 1
          if previous[j - 1]>=min: 
            store_found_repeat(protein, res, i-1, j-1, previous[j-1])
        else: #extend repeat
          current[j] = previous[j - 1] + 1 
      elif previous[j - 1]>=min:  #end repeat due mismatch
          store_found_repeat(protein, res, i-1, j-1, previous[j-1])

    if previous[y - 1]>=min:
        store_found_repeat(protein, res, i-1, y-1, previous[y-1])

    previous = current.copy()
    current = np.zeros(shape=y, dtype=np.int32)
  return res


def FAIR_algorithm_sensitive_to_occurences(protein: str, min: int, allow_overlapping=False) -> Dict[str, Set[Tuple[int, int]]]:
  #Based on https://core.ac.uk/download/pdf/291560955.pdf
  assert type(protein) == str, "protein must be a string"
  res_longest =  defaultdict(set)
  res_all = defaultdict(set)
  y = len(protein)
  previous = np.zeros(shape=y, dtype=np.int32)
  current = np.zeros(shape=y, dtype=np.int32)

  for i in range(y):
    for j in range(i + 1, y):
      if protein[i] == protein[j]:
        first,second = create_indicies_pairs(i, j, previous[j-1]+1) 
        if overlapping(first, second) and not allow_overlapping:#end repeat due overlap
          current[j] = 1
          if previous[j - 1]>=min: 
            store_found_repeat(protein, res_longest, i-1, j-1, previous[j-1])#store cause we store all subsequences on the way
            store_found_repeat(protein, res_all, i-1, j-1, previous[j-1])#store cause it is the longest that not ovar lap
        else: #extend repeat
          current[j] = previous[j - 1] + 1 
      if previous[j - 1]>=min:
        store_found_repeat(protein, res_all, i-1, j-1, previous[j-1]) #store cause we store all subsequences on the way
        if protein[i] != protein[j]: #end repeat due mismatch
          store_found_repeat(protein, res_longest, i-1, j-1, previous[j-1])

    if previous[y - 1]>=min:
        store_found_repeat(protein, res_longest, i-1, y-1, previous[y-1])#store cause we store all subsequences on the way
        store_found_repeat(protein, res_all, i-1, y-1, previous[y-1])#store cause it is the longest repeat (end of sequence)

    previous = current.copy()
    current = np.zeros(shape=y, dtype=np.int32)

  merged_dicts= choose_repeats_for_each_repeat_key(res_longest, res_all)
  related_repeats = get_related_repeats(merged_dicts)
  return merged_dicts, related_repeats

def choose_repeats_for_each_repeat_key(res_longest, res_all):
  merged_dict = deepcopy(res_longest)
  for repeat_key_longest, intervals_longest in res_longest.items():
      intervals_all = res_all[repeat_key_longest]
      if not intervals_longest.issubset(intervals_all):
        raise ValueError(f"Inconsistency detected for {repeat_key_longest}")
      if len(res_all[repeat_key_longest]) > len(intervals_longest):
        merged_dict[repeat_key_longest] = res_all[repeat_key_longest]

  return merged_dict

def get_related_repeats(repeats_dict):
    related_repeats = defaultdict(set)
    repeat_keys = list(repeats_dict.keys())

    for i, repeat1 in enumerate(repeat_keys):
        for j in range(i + 1, len(repeat_keys)):  # Ensures each pair is checked only once
            repeat2 = repeat_keys[j]

            # Check substring
            if repeat1 in repeat2:
                related_repeats[repeat1].add((repeat2,"substring"))
                related_repeats[repeat2].add((repeat1, "contains"))
                continue  # Skip overlap check if already related by substring
            if repeat2 in repeat1:
                related_repeats[repeat2].add((repeat1,"substring"))
                related_repeats[repeat1].add((repeat2, "contains"))
                continue  # Skip overlap check if already related by substring

            # Check overlapping intervals
            intervals1 = repeats_dict[repeat1]
            intervals2 = repeats_dict[repeat2]

            if any(overlapping(int1, int2) for int1 in intervals1 for int2 in intervals2):
                related_repeats[repeat1].add((repeat2, "overlapping"))
                related_repeats[repeat2].add((repeat1, "overlapping"))

    return related_repeats

        
        