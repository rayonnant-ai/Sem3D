# Sem3D

Semantic 3D tokenizer that converts text into a lossless coordinate stream of **(Tag, Quality)** pairs.

- **Tags** represent semantic identity: `P#` (persons), `A#` (actions), `O#` (objects), `D#` (destinations), `V#` (variables), `F#` (functions), `T#` (types)
- **Qualities** represent surface-form state: `BASE`, `PAST`, `DEF`, `INDEF`, `KW`, `OP`, `STR`, etc. (34 total)

Supports English prose (via spaCy) and source code (Python, Bash, C).

## Example

```
Input:  "Ben walked to the store."

Dictionary:
  P1  Ben    Entity
  A1  walk   Action
  D1  store  Destination

Stream:
  (P1, BASE) (A1, PAST) ('to', PREP) (D1, DEF, modifier="the") ('.', PUNCT)
```

The coordinate stream roundtrips losslessly back to the original text.

## 1D Encoding

The `Vocab` class flattens the 2D coordinate stream into a single integer token sequence (16,384 vocab size) for use with standard language models:

```python
from sem3d_tokenizer import Sem3DTokenizer, Vocab

tok = Sem3DTokenizer()
tag_dict, stream = tok.encode("Ben walked to the store.")

vocab = Vocab.build([stream])
ids = vocab.encode_1d(stream)     # [283, 0, 284, 12, ...]
decoded = vocab.decode_1d(ids)    # back to Coordinate stream
```

## Code Tokenization

```python
from sem3d_tokenizer import CodeTokenizer

code_tok = CodeTokenizer()
tag_dict, stream = code_tok.encode("def fib(n):\n    return n\n", lang="python")
print(code_tok.decode(tag_dict, stream))  # lossless roundtrip
```

Supported languages: `python`, `bash`, `c`.

## Setup

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

spaCy is only required for English prose tokenization. Code tokenization uses only the standard library.
