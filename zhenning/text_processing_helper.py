# return a list of words in a sentance 
def tokenizer(text, remove_special_character=True):
    text = text.lower()
    tokens = text.split(' ')
    if remove_special_character:
        for i in range(len(tokens)):
            s = ''.join(e for e in tokens[i] if e.isalnum())
            tokens[i] = s
    return tokens


# return the index of element
# return -1 if word not found
def search_word(tokens, word):
    word = word.lower()
    index = -1
    try:
        index = tokens.index(word)
    except:
        pass
    return index


# built on top of tokenizer and search_word
# pass in list of texts and a word
# return a dict. key is the keyword, and val is list of index(texts that contain the word)
def word_appearance(texts, k, d={}):
    d[k] = []
    for i in range(len(texts)):
        temp = tokenizer(texts[i])
        temp = list(set(temp))
        f = search_word(temp, k)
        if f!=-1:
            d[k].append(i)
    return d


# search a phase
# pass in a string and a phase
def search_phase(text, phase):
    text = text.lower()
    phase = phase.lower()
    index = text.find(phase)
    try:
        # not t.isalnum()
        temp = text[index+len(phase)]
        if temp.isalnum() or text[index-1].isalnum():
            return -1
    except:
        pass
    return index


# similar to word_appearance
# built on top of search_phase
# pass in list of texts and a word or phase
# return a dict. key is the keyword, and val is list of index(texts that contain the word)
def phase_appearance(texts, p, d={}):
    d[p] = []
    for i in range(len(texts)):
        index = search_phase(texts[i], p)
        if index!=-1:
            d[p].append(i)
    return d