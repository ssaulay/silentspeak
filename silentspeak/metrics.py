def levenshtein(a, b):
  """Calculates the Levenshtein distance between a and b.
  The code was taken from: http://hetland.org/coding/python/levenshtein.py
  """
  n, m = len(a), len(b)
  if n > m:
    # Make sure n <= m, to use O(min(n,m)) space
    a, b = b, a
    n, m = m, n
  current = list(range(n + 1))
  for i in range(1, m + 1):
    previous, current = current, [i] + [0] * n
    for j in range(1, n + 1):
      add, delete = previous[j] + 1, current[j - 1] + 1
      change = previous[j - 1]
      if a[j - 1] != b[i - 1]:
        change = change + 1
      current[j] = min(add, delete, change)
  return current[n]


def decode_y(y, batch_size, pred = True):
  decoded = y
  if pred:
    decoded = tf.keras.backend.ctc_decode(y, [MAX_LEN]*batch_size, greedy=False)[0][0].numpy()
  tf_bytes = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded]
  result = [t.numpy().decode() for t in tf_bytes]
  return result


def WER(y_true, y_pred, batch_size):
  #compute CTC decoded of predictions
  decode_y_true = decode_y(y_true,batch_size,False)
  decode_y_pred = decode_y(y_pred, batch_size, True)

  #compute WER
  total_wer = 0
  total_tokens = 0
  for gt, p in zip(decode_y_true, decode_y_pred):
    total_wer += levenshtein(gt.split(), p.split())
    total_tokens += len(gt.split())

  return total_wer/total_tokens


def CER(y_true, y_pred, batch_size):
  #compute CTC decoded of predictions
  decode_y_true = decode_y(y_true, batch_size,False)
  decode_y_pred = decode_y(y_pred, batch_size, True)

  #compute WER
  total_cer = 0
  total_chars = 0
  for gt, p in zip(decode_y_true, decode_y_pred):
    total_cer += levenshtein(list(gt), list(p))
    total_chars += len(list(gt))

  return total_cer/total_chars
