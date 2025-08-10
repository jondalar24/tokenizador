# Demo de Tokenización: Word / Char / Subword

Este proyecto muestra, en consola, cómo una misma frase se tokeniza con tres enfoques distintos:

- **Word-based** (NLTK)
- **Character-based** (carácter a carácter)
- **Subword-based** (BERT WordPiece) + **tokens especiales** + **vocabulario** + **IDs** (+ padding demostrativo)

Está pensado para acompañar el artículo de LinkedIn: *“Tokenización: el idioma que entienden los LLMs”*.

---

##  1) Preparar entorno virtual

### Opción A · `venv` (recomendada)
#### Windows (PowerShell)
```powershell
# En la carpeta del proyecto
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# Actualiza pip (opcional pero recomendado)
python -m pip install --upgrade pip
```

#### Linux / macOS (bash/zsh)
```bash
# En la carpeta del proyecto
python3 -m venv .venv
source .venv/bin/activate

# Actualiza pip (opcional)
python -m pip install --upgrade pip
```

> Para salir del entorno virtual: `deactivate`

### Opción B · Conda (alternativa)
```bash
conda create -n nlp311 python=3.11 -y
conda activate nlp311
```

---

##  2) Instalar dependencias con `requirements.txt`

Crea un archivo `requirements.txt` con el siguiente contenido (CPU‑only, versiones compatibles entre sí):

```txt
torch==2.2.2
torchtext==0.17.2
transformers==4.42.1
sentencepiece
nltk
```

Instala:

```bash
pip install -r requirements.txt
```

> 💡 La primera vez que ejecutes el script, `transformers` descargará automáticamente el tokenizador de BERT (WordPiece).  
> 💡 Si NLTK pide recursos, el script intentará descargar `punkt` automáticamente. Si tu red bloquea descargas, consulta “Solución de problemas” abajo.

---

##  3) Ejecutar

```bash
python tokenizacion.py
```

1) La pantalla se limpia y verás un **prompt**:
```
 Demo de tokenización (Word / Char / Subword)

Escribe una frase y pulsa Enter:
```

2) Escribe tu frase (en español o inglés) y pulsa Enter.

3) Verás tres bloques con la tokenización y un resumen final.

---

##  4) ¿Qué verás en la salida?

### A) Word-based (NLTK)
- Divide por **palabras y puntuación**.
- Preserva el significado semántico básico.
- Limitación: vocabulario enorme y problemas con **OOV** (palabras no vistas).

**Ejemplo:**
```
Word-based (NLTK)
Tokens: ['HAy', 'que', 'entender', 'la', 'tokenización', 'correctamente']
```

### B) Character-based
- Cada **carácter** (incluidos espacios) es un token.
- Vocabulario mínimo; **no hay OOV**.
- Limitación: secuencias muy largas y poca semántica.

**Ejemplo:**
```
Character-based
Tokens: ['H','A','y',' ', 'q','u','e',' ', ...]
```

### C) Subword-based (BERT WordPiece) + especiales + vocab
- **Palabras frecuentes** se mantienen enteras.
- **Palabras raras** se dividen en **subpalabras** (`##` indica sufijo que se une al token anterior).
- Se añaden **tokens especiales**:
  - `<bos>` (inicio), `<eos>` (fin), `<pad>` (relleno), `<unk>` (desconocido)
- Se construye un **vocabulario** (demo local) y se muestran **IDs** (token→entero).
- *Este es el enfoque de los LLM modernos.*

**Ejemplo:**
```
Subword-based (BERT WordPiece) + especiales + vocab
Tokens: ['<bos>', 'hay', 'que', 'en', '##ten', '##der', 'la', 'token', '##iza', '##cion', 'correct', '##ament', '##e', '<eos>']
IDs   : [2, 12, 14, 11, 9, 6, 13, 15, 8, 5, 10, 4, 7, 3]

ID | Token
---+-------
0  | <unk>
1  | <pad>
2  | <bos>
3  | <eos>
... etc.
```

### D) Padding demostrativo
- Muestra cómo **igualar longitudes** de secuencias con `<pad>` (útil para “hacer batches”).
- Ejemplo visual de **preprocesamiento** típico antes de entrenar/inferir.

---

##  5) Cómo encaja con la teoría

- **Word-based**: intuitivo pero con vocabulario enorme; problemas con plurales/derivaciones.
- **Character-based**: sin OOV, pero semántica débil y secuencias largas.
- **Subword-based**: equilibrio ideal → reduce OOV, mantiene significado morfológico.  
  - **BERT/WordPiece** usa `##sufijo`.  
  - Otros como **XLNet/T5** usan SentencePiece (`▁` para espacios).

En **PyTorch/torchtext**, tras tokenizar:
1) generamos **vocabularios** (token→ID),
2) añadimos **tokens especiales**,
3) y opcionalmente **paddeamos** las secuencias.

---

##  6) Solución de problemas

- **Windows / Visual C++**: si PyTorch diera errores de DLL, instala/repara  
  *Microsoft Visual C++ Redistributable 2015–2022 (x64)*.
- **NLTK pide recursos** (`punkt` / `punkt_tab`): el script descarga `punkt` automáticamente.  
  Si tu red bloquea descargas, ejecuta manualmente:
  ```python
  import nltk
  nltk.download('punkt')
  # (si te lo pidiera) nltk.download('punkt_tab')
  ```
- **Descargas de modelos** (`transformers`): la primera ejecución puede tardar más; requiere conexión a internet.

---

##  7) Estructura del repo (sugerida)

```
.
├── tokenizacion.py     # Script principal de demo
├── requirements.txt    # Dependencias
└── README.md           # Este archivo
```
