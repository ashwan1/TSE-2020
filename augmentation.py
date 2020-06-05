from typing import List

from nltk.corpus import wordnet, stopwords
import random

_stop_words = stopwords.words('english')

_keyboard_errors = {
    "1": ["!", "2", "@", "q", "w"],
    "2": ["@", "1", "!", "3", "#", "q", "w", "e"],
    "3": ["#", "2", "@", "4", "$", "w", "e"],
    "4": ["$", "3", "#", "5", "%", "e", "r"],
    "5": ["%", "4", "$", "6", "^", "r", "t", "y"],
    "6": ["^", "5", "%", "7", "&", "t", "y", "u"],
    "7": ["&", "6", "^", "8", "*", "y", "u", "i"],
    "8": ["*", "7", "&", "9", "(", "u", "i", "o"],
    "9": ["(", "8", "*", "0", ")", "i", "o", "p"],
    "!": ["@", "q"],
    "@": ["!", "#", "q", "w"],
    "#": ["@", "$", "w", "e"],
    "$": ["#", "%", "e", "r"],
    "%": ["$"],
    "q": ["1", "!", "2", "@", "w", "a", "s"],
    "w": ["1", "!", "2", "@", "3", "#", "q", "e", "a", "s", "d"],
    "e": ["2", "@", "3", "#", "4", "$", "w", "r", "s", "d", "f"],
    "r": ["3", "#", "4", "$", "5", "%", "e", "t", "d", "f", "g"],
    "t": ["4", "$", "5", "%", "6", "^", "r", "y", "f", "g", "h"],
    "y": ["5", "%", "6", "^", "7", "&", "t", "u", "g", "h", "j"],
    "u": ["6", "^", "7", "&", "8", "*", " t", "i", "h", "j", "k"],
    "i": ["7", "&", "8", "*", "9", "(", "u", "o", "j", "k", "l"],
    "o": ["8", "*", "9", "(", "0", ")", "i", "p", "k", "l"],
    "p": ["9", "(", "0", ")", "o", "l"],
    "a": ["q", "w", "a", "s", "z", "x"],
    "s": ["q", "w", "e", "a", "d", "z", "x", "c"],
    "d": ["w", "e", "r", "s", "f", "x", "c", "v"],
    "f": ["e", "r", "t", "d", "g", "c", "v", "b"],
    "g": ["r", "t", "y", "f", "h", "v", "b", "n"],
    "h": ["t", "y", "u", "g", "j", "b", "n", "m"],
    "j": ["y", "u", "i", "h", "k", "n", "m", ",", "<"],
    "k": ["u", "i", "o", "j", "l", "m", ",", "<", ".", ">"],
    "l": ["i", "o", "p", "k", ";", ":", ",", "<", ".", ">", "/", "?"],
    "z": ["a", "s", "x"],
    "x": ["a", "s", "d", "z", "c"],
    "c": ["s", "d", "f", "x", "v"],
    "v": ["d", "f", "g", "c", "b"],
    "b": ["f", "g", "h", "v", "n"],
    "n": ["g", "h", "j", "b", "m"],
    "m": ["h", "j", "k", "n", ",", "<"]
}


def introduce_keyboard_error(words: List[str], n: int) -> (List[str], dict):
    change_log = {}
    new_words: List[str] = words.copy()
    for i in range(n):
        word_idx = random.choice(range(len(words)))
        char_idx = random.choice(range(len(words[word_idx])))
        new_words[word_idx] = new_words[word_idx].replace(words[word_idx][char_idx],
                                                          random.choice(_keyboard_errors.get(words[word_idx][char_idx],
                                                                                             words[word_idx][
                                                                                                 char_idx])))
        change_log[words[word_idx]] = new_words[word_idx]
    return new_words, change_log


def random_char_deletion(words: List[str]) -> (List[str], dict):
    try:
        random_word: str = random.choice([word for word in words if word not in _stop_words])
        random_char = random.choice(random_word)
        new_random_word = random_word.replace(random_char, '', 1)
        new_words = ' '.join(words).replace(random_word, new_random_word, 1).split()
        change_log = {random_word: new_random_word}
        return new_words, change_log
    except Exception:
        return words, {}


def random_char_repeat(words: List[str]) -> (List[str], dict):
    try:
        random_word: str = random.choice([word for word in words if word not in _stop_words])
        random_char = random.choice(random_word)
        new_random_word = random_word.replace(random_char, random_char * 2, 1)
        new_words = ' '.join(words).replace(random_word, new_random_word, 1).split()
        change_log = {random_word: new_random_word}
        return new_words, change_log
    except Exception:
        return words, {}


def synonym_replacement(words: List[str], n: int) -> (List[str], dict):
    change_log = {}
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in _stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = _get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            # print("replaced", random_word, "with", synonym)
            change_log[random_word] = synonym
            num_replaced += 1
        if num_replaced >= n:  # only replace up to n words
            break
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')
    return new_words, change_log


def _get_synonyms(word: str) -> List[str]:
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


if __name__ == '__main__':
    sent = 'IT is so annoying when she starts typing on her computer in the middle of the night!'.lower().split()
    print(random_char_deletion(sent))
    print('=' * 10)
    print(random_char_repeat(sent))
