# Helper function to compare two nested Dicts.
# Note that this function only ensures all the fields in dict_a have definition
# and same value in dict_b. This function does not guarantee that
# dict_a == dict_b.
def nested_dict_compare(dict_a, dict_b):
  for k, v in sorted(dict_a.items()):
    if k not in dict_b:
      return False
    if isinstance(v, dict) and isinstance(dict_b[k], dict):
      if not nested_dict_compare(dict_a[k], dict_b[k]):
        return False
    else:
      # A caveat: When dict_a[k] = 1, dict_b[k] = True, the return is True.
      if dict_a[k] != dict_b[k]:
        return False
  return True