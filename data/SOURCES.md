# Training Data Sources

`prepare_corpus.py` builds the deterministic training text used by the local
commands and the GitHub Actions workflow. It strips Project Gutenberg framing,
samples long books across their length, repeats the grammar corpus, removes
duplicate chat turns, and changes the generated chat identity from Ultron to
Eto.

The Project Gutenberg texts are public-domain works in the United States:

| File | Work | Project Gutenberg eBook |
|---|---|---:|
| `alice.txt` | Alice's Adventures in Wonderland | 11 |
| `drac.txt` | Dracula | 345 |
| `earnest.txt` | The Importance of Being Earnest | 844 |
| `frank.txt` | Frankenstein | 84 |
| `ge.txt` | Great Expectations | 1400 |
| `jane.txt` | Jane Eyre | 1260 |
| `moby.txt` | Moby-Dick | 2701 |
| `pp.txt` | Pride and Prejudice | 1342 |
| `sherlock.txt` | The Adventures of Sherlock Holmes | 108 |
| `tom.txt` | The Adventures of Tom Sawyer | 74 |
| `wp.txt` | War and Peace | 2600 |

`grammar.txt` and `chat.txt` are supplied/generated local corpora;
`english_grammar.txt` is an original starter text. The builder intentionally
excludes `mc.txt` and `mc_qa.txt`: their headers identify Minecraft Wiki content
under CC BY-NC-SA 3.0, which is not included in this general English training
run. `eto-vocab.txt` is a token-ID list, not prose, and is also excluded. The
source files are left unchanged.