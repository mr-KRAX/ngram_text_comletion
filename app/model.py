import heapq
from typing import Tuple, Union, List
from collections import Counter, defaultdict
import tqdm


class PrefixTreeNode:
    def __init__(self):
        # словарь с буквами, которые могут идти после данной вершины
        self.children: dict[str, PrefixTreeNode] = {}
        self.is_end_of_word = False


class PrefixTree:
    def __init__(self, vocabulary: List[str]):
        """
        vocabulary: список всех уникальных токенов в корпусе
        """
        self.root = PrefixTreeNode()
        for word in vocabulary:
            self.insert(word)

    def insert(self, word: str):
        """
        Inserts a word into the prefix tree.
        """
        current_node = self.root
        for char in word:
            # If character not in current node's children, add it
            if char not in current_node.children:
                current_node.children[char] = PrefixTreeNode()
            # Move to the next node
            current_node = current_node.children[char]
        # Mark the end of a word
        current_node.is_end_of_word = True

    def search_prefix(self, prefix: str) -> List[str]:
        """
        Возвращает все слова, начинающиеся на prefix
        prefix: str – префикс слова
        """

        result = []
        current_node = self.root

        for char in prefix:
            if char in current_node.children:
                current_node = current_node.children[char]
            else:
                return result

        def collect_all_words(node, path):
            if node.is_end_of_word:
                result.append(path)
            for char, child_node in node.children.items():
                collect_all_words(child_node, path + char)

        collect_all_words(current_node, prefix)

        return result


class WordCompletor:
    def __init__(self, corpus: List[str]):
        """
        corpus: list – корпус текстов
        """
        counter = Counter()
        for words in tqdm.tqdm(corpus, desc="init WordCompletor"):
            if corpus:
                counter.update(words)

        total = sum(counter.values())
        self.probs = {w: p/total for w, p in counter.items()}
        self.vocabulary = counter.keys()

        self.prefix_tree = PrefixTree(self.vocabulary)

    def get_words_and_probs(self, prefix: str) -> Tuple[List[str], List[float]]:
        # Find all words starting with the given prefix
        words = self.prefix_tree.search_prefix(prefix)
        probs = [self.probs[word] for word in words]
        return words, probs


class NGramLanguageModel:
    def __init__(self, corpus, n):
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.n = n

        for words in tqdm.tqdm(corpus, desc="init NGramLanguageModel"):
            for i in range(len(words)):
                for j in range(i+1, min(len(words), i+n+1)):
                    context = tuple(words[i:j])
                    word = words[j]
                    self.ngram_counts[context][word] += 1
                    self.context_counts[context] += 1



    def get_next_words_and_probs(self, prefix: list) -> Tuple[List[str], List[float]]:
        """
        Возвращает список слов, которые могут идти после prefix,
        а так же список вероятностей этих слов
        """
        context = tuple(prefix[-self.n:])
        next_words_counter = self.ngram_counts.get(context, Counter())

        if not next_words_counter:
            return [], []

        total_count = self.context_counts[context]
        next_words = list(next_words_counter.keys())
        probs = [count / total_count for count in next_words_counter.values()]

        return next_words, probs


def get_top_n_words(words: List[str], probs: List[float], n: int) -> List[str]:
    top_n = heapq.nlargest(n, zip(probs, words))
    top_n_words = [word for _, word in top_n]
    return top_n_words


def get_top_word(words: List[str], probs: List[float]) -> str:
    max_p, max_w = 0, None
    for w, p in zip(words, probs):
        if p > max_p:
            max_p, max_w = p, w
    return max_w


class TextSuggestion:
    def __init__(self, word_completor, n_gram_model):
        self.word_completor = word_completor
        self.n_gram_model = n_gram_model

    def get_corrections(self, prefix: str, n=3) -> Tuple[List[str], List[float]]:
        """
        Возвращает n самых вероятных дополнений слова
        """
        return get_top_n_words(*self.word_completor.get_words_and_probs(prefix), n)

    def suggest_text(self, text: Union[str, list], n_words=3, need_correction=True, n_texts=1) -> list[list[str]]:
        """
        Возвращает возможные варианты продолжения текста (по умолчанию только один)
        
        text: строка или список слов – написанный пользователем текст
        n_words: число слов, которые дописывает n-граммная модель
        need_correction: нужно ли дополнять последнее слово
        n_texts: число возвращаемых продолжений (пока что только одно)
        
        return: list[list[srt]] – список из n_texts списков слов, по 1 + n_words слов в каждом
        Первое слово – это то, которое WordCompletor дополнил до целого.
        """
        if isinstance(text, str):
            text = text.split()

        if not text:
            return []

        suggestions = []
        last_word = text[-1]
        extended_text = text
        if need_correction:
            completion = get_top_word(*self.word_completor.get_words_and_probs(last_word))
            if completion:
                extended_text = text[:-1] + [completion]
                last_word = completion

        next_words = self.n_gram_model.get_next_words_and_probs(extended_text)
        next_words = get_top_n_words(*next_words, n_texts)
        if next_words == []:
            return []

        for n_w in next_words:
            new_ext_text = extended_text + [n_w]
            current_suggestion = [last_word, n_w]
            for _ in range(n_words-1):
                words = self.n_gram_model.get_next_words_and_probs(new_ext_text)
                next_word = get_top_word(*words)
                current_suggestion.append(next_word)
                extended_text = extended_text[1:] + [next_word]
            suggestions.append(current_suggestion)

        return suggestions
