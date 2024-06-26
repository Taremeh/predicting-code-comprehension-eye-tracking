class LruCache(object):

    def __init__(self, capacity=3000):
        self.capacity = capacity
        self.cache = collections.OrderedDict()
        self.lock = threading.Lock()
        self.running = True

    def get(self, key):
        with self.lock:
            record = None
            try:
                record = self.cache.pop(key)
                self.cache[key] = record
            except KeyError:
                pass
            return record

    def set(self, key, record):
        with self.lock:
            try:
                self.cache.pop(key)
            except KeyError:
                if len(self.cache) >= self.capacity:
                    self.cache.popitem(last=False)

            self.cache[key] = record

    def __str__(self):
        out_str = ""
        for key, value in list(self.cache.items()):
            if isinstance(value, str):
                out_str += " %s => %s<br>\n" % (key, value)
            elif isinstance(value, dict) or isinstance(value, list):
                out_str += " %s => %s<br>\n" % (key, json.dumps(value))

        return out_str