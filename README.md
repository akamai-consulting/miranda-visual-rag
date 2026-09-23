# Miranda Visual RAG

A small, local-first experiment in **multimodal retrieval**: turn an image collection into CLIP embeddings, keep the embeddings in a PyTorch tensor, and retrieve visually relevant images with direct matrix multiplication (GEMM)—without requiring FAISS or a vector database.

Miranda grew from a simple question: **what information are users actually trying to retrieve, and how should that information be represented?** For a fashion archive, some of that information lives in text and metadata; some lives in the pixels themselves.

## Architecture

```text
Fashion archive
      |
      v
  OpenCLIP
      |
      v
Normalized image embeddings
      |
      v
 PyTorch tensor

User query (English / multilingual)
      |
      v
Language / domain routing
      |------------------------------|
      |                              |
simple English                 multilingual / ambiguous
      |                              |
      |                         local LLM
      |                              |
      |<----- concise English intent-|
      v
CLIP text encoder
      |
      v
normalized query vector
      |
      v
PyTorch GEMM: q @ E.T
      |
      v
Top-K images + confidence gate
```

The same embedding space also makes **image-to-image retrieval** a natural extension: encode an uploaded image with CLIP's image encoder and search the same matrix.

## Why no vector database?

This repository is an experiment, not an argument that vector databases are unnecessary. The goal is to **measure before adding infrastructure**.

In the experiment described in the accompanying Akamai blog post, the corpus reached **289,222 images**, represented as 512-dimensional FP32 vectors. The embedding matrix was roughly 565 MiB, and direct PyTorch GEMM remained fast on Apple Silicon. At larger scales—or when distributed storage, metadata filtering, frequent updates, durability, multi-tenancy, or operational requirements dominate—a vector database may be the better architecture.

## Dataset

The large test corpus used for the experiment came from the **DeepFashion Category and Attribute Prediction Benchmark**.

**DeepFashion is not included in this repository.** Download it from the official project source and comply with its applicable license/terms. Then configure `IMAGE_ROOT` (or the command-line argument used by the indexing script) to point at your local image directory.

Official project: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html

The original experiment contained **289,219** DeepFashion JPG files. Three personally photographed Akamai × UNIQLO PEACE FOR ALL T-shirt images were then added, producing the **289,222-image** corpus discussed in the article.

## Example Akamai × UNIQLO images

`examples/akamai-peace-for-all/` is reserved for the three photographs used in the final retrieval experiment.

These photographs were taken by the repository author of a personally owned T-shirt. The photographs may be licensed separately from the software; see the README in that directory. Akamai, UNIQLO, PEACE FOR ALL, associated logos, and the depicted garment design remain the property of their respective rights holders.

The final test query was intentionally generic:

```text
t-shirt with nerdy computer code
```

It did **not** contain the word `Akamai`.

## Quick start

### 1. Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Provide an image collection

For a small reproducible test, create a directory such as:

```text
data/images/
  jacket_001.jpg
  shirt_002.jpg
  ...
```

For the full DeepFashion experiment, download DeepFashion separately from its official source.

### 3. Build embeddings

```bash
python build_index.py --image-root data/images --output index.pt
```

### 4. Run Miranda

```bash
streamlit run app.py
```

> The public release should keep model names, paths, thresholds, and LLM endpoints configurable. Do not commit API keys or internal endpoints.

## Retrieval math

If `q` is a normalized query embedding and `E` is the matrix of normalized image embeddings, similarity is simply:

```text
S = q E^T
```

The highest values in `S` are the Top-K candidates.

## Routing principle

Miranda deliberately does **not** send every query through an LLM.

- Straightforward English fashion query → CLIP directly
- Multilingual or ambiguous fashion query → language model normalization → CLIP
- Out-of-domain request → reject rather than becoming a general-purpose assistant
- Retrieval below the configured confidence threshold → do not present it as a confident match

This separation gives each component a narrow job: the LLM interprets language, CLIP represents visual meaning, and GEMM performs retrieval.

## Security-by-design test cases

A useful regression set includes both sides of the domain boundary:

```text
Write shell code.                         # reject
Write a C++ Hello World program.          # reject
Find me a T-shirt with shell code.        # accept
black winter coat                         # accept, direct CLIP path
冬に着る暖かいコートを見せて              # accept, multilingual normalization
Mostrami un cappotto nero per l'inverno.  # accept, multilingual normalization
```

The point is not keyword blocking. `code` can be maliciously irrelevant in one context and completely legitimate fashion intent in another.

## Observability

During development it is useful to expose intermediate representations, for example:

```text
Original query: Mostrami un cappotto nero per l'inverno.
CLIP query:     black winter coat
Route:          multilingual -> LLM -> CLIP
```

This helps distinguish a retrieval failure from an upstream interpretation failure.

## Hardware

The prototype was developed on Apple Silicon using PyTorch MPS. Keep device selection portable so the same code can move to CUDA without redesigning the retrieval pipeline.

```python
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

## Repository scope

This repository should contain the **reproducible architecture**, not a redistribution of the research dataset or a snapshot of a developer workstation. Before publishing, verify that it contains no:

- credentials or API keys
- internal Akamai endpoints or confidential information
- customer data
- absolute local filesystem paths
- downloaded DeepFashion images or derived corpus artifacts that should not be redistributed
- large generated embedding indexes unless their licensing and distribution are intentional

## Related reading

The accompanying Akamai blog article, **“What Would RAG Look Like If Miranda Priestly Were the User?”**, explains the motivation, measurements, routing decisions, security experiments, and the final semantic retrieval test. Add the public URL here after publication.

## License

Software licensing is defined in `LICENSE`. Dataset and example-image rights are separate; see `DATASETS.md` and the example-image README.

## Acknowledgments

- OpenCLIP / CLIP ecosystem
- PyTorch
- Ollama and open-weight language models used during local prototyping
- DeepFashion, used as a research corpus under its applicable terms
- Akamai × UNIQLO PEACE FOR ALL, which provided the rather nerdy T-shirt that helped inspire the experiment
