# Training Data Sources

`prepare_corpus.py` builds the deterministic training text used by the local
commands and the GitHub Actions workflow. It strips Project Gutenberg framing,
samples long texts across their length, repeats the grammar corpus, removes
duplicate chat turns, extracts word entries from the tab-separated vocabulary,
and changes Ultron speaker/name references to Eto.

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
`english_grammar.txt` is an original starter text. `mc.txt` and `mc_qa.txt`
contain Minecraft Wiki material under CC BY-NC-SA 3.0. Their source files are
preserved unchanged, and the prepared corpus/model must be treated as
noncommercial share-alike material. See [MINECRAFT-NOTICE.md](MINECRAFT-NOTICE.md)
for attribution and usage terms. Obtain separate permission before any
commercial use. `eto-vocab.txt` is included only after token IDs and special
symbols are removed; its word list receives a single low-weight section.

## Conversation and story datasets (added for the generative chat model)

Downloaded by `data/fetch_datasets.py` and converted by `data/convert_external.py`:

- Stanford Alpaca, cleaned by gururise (CC BY-NC 4.0): short instruction/response turns.
- ChatterBot English corpus (gunthercox/chatterbot-corpus, BSD-3-Clause): small talk and bot profile.
- DailyDialog (Li et al., CC BY-NC-SA 4.0): everyday multi-turn conversations.
- TinyStories (Eldan and Li, CDLA-Sharing-1.0): simple, fluent English stories.
- Tiny Shakespeare (karpathy/char-rnn): plays.

Most are noncommercial; treat the trained model as noncommercial material.
