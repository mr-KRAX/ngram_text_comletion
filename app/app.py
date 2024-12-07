import json

import reflex as rx
from reflex import State

import app.model as m
from app.utils import SimpleLogger


CONTEXT_LEN = 4

logger = SimpleLogger()
logger.set_level(0)

with open('corpus.json', 'r') as f:
    corpus = json.load(f)
corpus = corpus[:100000]
logger.info('Corpus loaded: example:', corpus[0])

word_completer = m.WordCompletor(corpus)
logger.info('WordCompletor initiated')

ngram_model = m.NGramLanguageModel(corpus, n=CONTEXT_LEN)
logger.info('NGramLanguageModel initiated')

text_suggester = m.TextSuggestion(word_completer, ngram_model)
logger.info('TextSuggestion initiated')

log_counter = 1


def log_new_iteration():
    global log_counter
    log_counter += 1
    logger.debug(f'-------{log_counter}-------')


class State(rx.State):
    input_text = ''
    suggestions = []

    def on_sug_click(self, val):
       self.input_text = ' '.join(self.input_text.split()[:-1] + [val]) + ' '
       self.upd_suggestions(self.input_text)

    def upd_suggestions(self, new_text):
        log_new_iteration()

        self.input_text = new_text
        words = self.input_text.split()
        if not words:
            self.suggestions = []
            return

        need_complete = new_text[-1] != ' '

        words = words[-CONTEXT_LEN:]
        
        self.suggestions = []
        if need_complete:
            prefix = words[-1]
            completions = text_suggester.get_corrections(prefix, n=5)
            logger.debug(f'APP: prefix: {prefix} compls: {completions}')
            self.suggestions = completions

        logger.debug(f'QQQ: APP: words: {words}, new_text: {new_text}')

        def upd_suggestions(suggestions, expected_n):
            logger.debug('QQQ: APP: suggester result:', suggestions)
            if suggestions == [[None]]:
                return
            for suggestion in suggestions:
                if len(suggestion) < expected_n + 1:
                    continue
                if None in suggestion:
                    continue
                joined_suggestion = ' '.join(suggestion)
                if joined_suggestion in self.suggestions:
                    continue
                self.suggestions.append(joined_suggestion)
        
        suggestions = text_suggester.suggest_text(words, n_words=1, need_correction=need_complete, n_texts=3)
        upd_suggestions(suggestions,1)
        suggestions = text_suggester.suggest_text(words, n_words=2, need_correction=need_complete, n_texts=1)
        upd_suggestions(suggestions,2)


def index() -> rx.Component:
    return rx.container(
        rx.color_mode.button(position='top-right'),
        rx.center(
            rx.box(
                rx.heading('Any Suggestions?', size='7'),
                rx.divider(),
                rx.input(
                    value=State.input_text,
                    on_change=State.upd_suggestions,
                    placeholder='...',
                    max_length=100),
                rx.foreach(
                    State.suggestions,
                    lambda s: rx.text(s, on_click=lambda: State.on_sug_click(s))
                ),
                width='60%'
            ),
        )
    )


app = rx.App()
app.add_page(index)
