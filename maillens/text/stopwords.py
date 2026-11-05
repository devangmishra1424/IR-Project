"""English stop-word list (the classic NLTK list, apostrophe-free).

Tokens are split on non-alphanumerics, so "don't" becomes "don" + "t"; both halves are listed.
Kept in the repository so the pipeline needs no runtime download.
"""

# A whitespace-separated block is far easier to read and diff than 150 quoted strings.
_WORDS = """
    i me my myself we our ours ourselves you your yours yourself yourselves he him his himself
    she her hers herself it its itself they them their theirs themselves what which who whom
    this that these those am is are was were be been being have has had having do does did doing
    a an the and but if or because as until while of at by for with about against between into
    through during before after above below to from up down in out on off over under again
    further then once here there when where why how all any both each few more most other some
    such no nor not only own same so than too very s t can will just don should now d ll m o re
    ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn
    weren won wouldn
"""

ENGLISH_STOPWORDS: frozenset[str] = frozenset(_WORDS.split())
