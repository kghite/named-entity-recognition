class Reader():
    def __init__(self, filename):
        self.filename = "/data2/user_data/khite/ner_data/" + filename
        self.IGNORE_WORD = "-X-"

    def process_words(self):
        f = open(self.filename, "r+")
        words = []
        sentences = []
        for line in f.readlines():
            if len(line.strip()) > 0:
                word = self.process_line(line)
                if word.pos != self.IGNORE_WORD:
                    words.append(word)
            else:
                if len(words) > 0:
                    sentences.append(words)
                    words = []
        if len(words) > 0:
            sentences.append(words)
        return sentences

    def process_line(self, line):
        parts = line.strip().split()
        if len(parts) == 4:
            return TaggedWord(parts[0], parts[1], parts[2], parts[3])
        return TaggedWord("", self.IGNORE_WORD, "", "")

class TaggedWord():
    def __init__(self, word, pos, phrase, tag):
        self.word = word
        self.pos = pos
        self.phrase = phrase
        # TODO: Figure out how to handle boundaries
        self.tag = tag.split("-")[1] if tag.startswith("I-") or tag.startswith("B-") else tag

    def __str__(self):
        return "({}, {}, {}, {})".format(self.word, self.pos, self.phrase, self.tag)

# EXAMPLE
'''
tags = {}
r = Reader("eng.train")
lines = r.process_words()
for line in lines:
    for word in line:
        tags[word.tag] = 1

print tags.keys()
print len(tags.keys())
'''
