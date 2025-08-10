# --- Demo de tokenización: word / char / subword (con vocab y tokens especiales) ---

import os
import sys
import shutil
import textwrap
import nltk
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
from torchtext.vocab import build_vocab_from_iterator

# Descargar 'punkt' si no está (necesario para word_tokenize)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# -------- Utilidades de consola / formato --------
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def term_width(default=100):
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default

def box_title(title: str):
    w = term_width()
    bar = "─" * min(len(title), w)
    print(f"\n{title}\n{bar}")

def pretty_list(label: str, items, sep=" | ", indent=2):
    w = term_width()
    body = sep.join(items)
    wrapped = textwrap.fill(body, width=w, subsequent_indent=" " * indent)
    print(f"{label}: {wrapped}")

def print_table(headers, rows, padding=1):
    # Tabla simple sin dependencias
    cols = len(headers)
    widths = [len(h) for h in headers]
    for r in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(r[i])))
    def fmt_row(r):
        return " | ".join(str(r[i]).ljust(widths[i]) for i in range(cols))
    print(fmt_row(headers))
    print("-+-".join("-" * widths[i] for i in range(cols)))
    for r in rows:
        print(fmt_row(r))

# -------- 1) Word-based (NLTK) --------
def tokenize_words(text: str):
    # Si quieres evitar la descarga de modelos, usa preserve_line=True
    return word_tokenize(text, language="spanish")

# -------- 2) Character-based --------
def tokenize_chars(text: str):
    return list(text)

# -------- 3) Subword-based (BERT WordPiece) + torchtext vocab --------
def tokenize_subword_with_vocab(text: str):
    # a) Tokenización subword con BERT (WordPiece)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    subword_tokens = tokenizer.tokenize(text)

    # b) Añadir tokens especiales
    specials = ["<unk>", "<pad>", "<bos>", "<eos>"]
    seq = ["<bos>"] + subword_tokens + ["<eos>"]

    # c) Construir vocabulario con torchtext (para la demo, solo con esta secuencia)
    def iterator():
        yield seq

    vocab = build_vocab_from_iterator(iterator(), specials=specials)
    vocab.set_default_index(vocab["<unk>"])

    # d) Indexar
    ids = [vocab[token] for token in seq]
    return seq, ids, vocab

# -------- Padding demostrativo --------
def pad_to_max_length(seqs, pad_token="<pad>"):
    max_len = max(len(s) for s in seqs)
    return [s + [pad_token] * (max_len - len(s)) for s in seqs], max_len

# ================== Programa principal ==================
if __name__ == "__main__":
    try:
        clear_screen()
        print("Demo de tokenización (Word / Char / Subword)\n")
        text = input("Escribe una frase y pulsa Enter: ").strip()
        if not text:
            print("\nNo se ha introducido texto. Saliendo.")
            sys.exit(0)

        clear_screen()  # limpiar para que la salida quede “limpia”

        # 1) Word tokens
        word_tokens = tokenize_words(text)
        box_title("Word-based (NLTK)")
        pretty_list("Tokens", word_tokens)

        # 2) Char tokens
        char_tokens = tokenize_chars(text)
        box_title("Character-based")
        # Mostrar chars sin espacios o con espacios visibles (elige):
        # visibles = [c if c != " " else "␠" for c in char_tokens]
        pretty_list("Tokens", char_tokens)

        # 3) Subword + especiales + vocab + ids
        sub_tokens, sub_ids, vocab = tokenize_subword_with_vocab(text)
        box_title("Subword-based (BERT WordPiece) + especiales + vocab")
        pretty_list("Tokens", sub_tokens)
        pretty_list("IDs   ", [str(i) for i in sub_ids])

        # Vocab (token -> id) ordenado por id
        itos = vocab.get_itos()  # id -> token (lista)
        rows = [(i, itos[i]) for i in range(len(itos))]
        print()
        print_table(["ID", "Token"], rows)

        # Padding demostrativo
        padded_seqs, max_len = pad_to_max_length(
            [word_tokens, char_tokens, sub_tokens], pad_token="<pad>"
        )
        box_title(f"Padding demostrativo (longitud objetivo = {max_len})")
        pretty_list("Word-based   padded", padded_seqs[0])
        pretty_list("Character     padded", padded_seqs[1])
        pretty_list("Subword-based padded", padded_seqs[2])

        # Resumen final
        box_title("Resumen")
        print_table(
            ["Vista", "Nº tokens"],
            [
                ("Word-based", len(word_tokens)),
                ("Character-based", len(char_tokens)),
                ("Subword-based", len(sub_tokens)),
            ],
        )

    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
