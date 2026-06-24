# PLN-RAG

A REST API service for Probabilistic Logic Network (PLN) based retrieval-augmented reasoning.
Ingests natural language text, converts it to PLN atoms via a pluggable semantic parser,
stores facts in a PeTTaChainer atomspace, and answers questions via logical proof.

Current service assumptions:

- standalone self-hosted service
- one shared AtomSpace / knowledge base
- no multi-user isolation or tenant separation

## Architecture

```
Text → Chunker → SemanticParser → PeTTaChainer (atomspace + reasoning) → AnswerGenerator → Response
                      ↑
              Qdrant context retrieval
                      ↑
           Ollama (runs on host machine)
```

## Prerequisites

Ollama runs on your **host machine**, not inside Docker. Install it and pull the embedding model before starting the service:

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the embedding model (once — persists in ~/.ollama)
ollama pull nomic-embed-text

# Verify it is running
curl http://localhost:11434    # should return "Ollama is running"
```

## Quick start (Docker)

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY
# OLLAMA_URL can stay as localhost in .env; docker-compose overrides it for containers
# PLNRAG_PARSER controls the parser inside Docker Compose

docker compose up --build
```

The API will be available at http://localhost:8000.
Interactive docs at http://localhost:8000/docs.

> **Linux note:** `host.docker.internal` is not automatically available on Linux.
> The `docker-compose.yml` already includes `extra_hosts: host.docker.internal:host-gateway`
> to handle this. No manual action needed.

## API endpoints

### POST /ingest
Ingest texts into the knowledge base.
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts": ["People who eat fish are smart.", "Kebede eats fish."]}'
```

### POST /reason
Run a reasoning request against the knowledge base.
```bash
curl -X POST http://localhost:8000/reason \
  -H "Content-Type: application/json" \
  -d '{"query": "Is Kebede smart?"}'
```

### DELETE /reset
Clear the knowledge base (fully or partially).
```bash
# Clear everything
curl -X DELETE http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"scope": "all"}'

# Clear only vector DB (re-index without losing atomspace)
curl -X DELETE http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"scope": "vectordb"}'
```

### GET /health
```bash
curl http://localhost:8000/health
```

### GET /ready
```bash
curl http://localhost:8000/ready
```

## Switching parsers

Set `PARSER` in `.env` for local runs. For Docker Compose, use `PLNRAG_PARSER`
to avoid accidental overrides from a shell-level `PARSER` variable.

```bash
# Use NL2PLN (DSPy-based, SIMBA/GEPA optimized)
PARSER=nl2pln
PLNRAG_PARSER=nl2pln
NL2PLN_MODULE_PATH=data/simba_all.json

# Use CanonicalPLN parser (separate tuned SIMBA artifact)
PARSER=canonical_pln
PLNRAG_PARSER=canonical_pln
CANONICAL_PLN_NL2PLN_MODULE_PATH=data/simba_canonical_pln.json

# Use Manhin's parser (format self-correction + FAISS predicate store)
PARSER=manhin
PLNRAG_PARSER=manhin
```

`nl2pln` and `canonical_pln` intentionally use separate compiled artifacts so baseline
comparisons stay clean. Tune `data/simba_canonical_pln.json` without modifying the
baseline `simba_all.json`.

Query fallback execution can be toggled independently at runtime:

```bash
QUERY_FALLBACK_ENABLED=true
```

When disabled, the service runs only the original generated query. When enabled,
it may try later fallback candidates produced by the parser.

## ConceptNet Background Knowledge

ConceptNet can be loaded as readonly background knowledge and indexed into the
same Qdrant collection as normal ingested facts.

The repository tracks generated ConceptNet runtime artifacts so ConceptNet can be
enabled without committing the raw ConceptNet dump:

- `data/conceptnet/conceptnet_background.metta`
- `data/conceptnet/conceptnet_background.jsonl`
- `data/conceptnet/conceptnet_manifest.json`

The raw dump `data/conceptnet/conceptnet-assertions-5.7.0.csv.gz` is intentionally
ignored because it is large. It is only needed when rebuilding the generated
artifacts.

```bash
CONCEPTNET_ENABLED=true
CONCEPTNET_AUTOLOAD=true
CONCEPTNET_ATOMSPACE_PATH=data/conceptnet/conceptnet_background.metta
CONCEPTNET_VECTOR_PAYLOAD_PATH=data/conceptnet/conceptnet_background.jsonl
CONCEPTNET_COVERAGE_PERCENT=100.0
CONCEPTNET_SAMPLE_SEED=42
CONCEPTNET_AUTO_REBUILD_ON_CHANGE=false
```

Background points use the normal `nl` / `pln` payload schema with extra metadata
such as `source=conceptnet` and `background=true`.

To rebuild the aligned background files from a raw ConceptNet dump, first place
the dump at `data/conceptnet/conceptnet-assertions-5.7.0.csv.gz`, then run:

```bash
cp "/path/to/conceptnet-assertions-5.7.0.csv.gz" data/conceptnet/
python scripts/conceptnet/export_conceptnet.py
```

Set `CONCEPTNET_AUTO_REBUILD_ON_CHANGE=true` only on machines that have the raw
dump available and should regenerate artifacts automatically when the ConceptNet
export settings change.

Coverage is controlled at export time. For example:

```bash
CONCEPTNET_COVERAGE_PERCENT=1.0
CONCEPTNET_SAMPLE_SEED=42
python scripts/conceptnet/export_conceptnet.py
```

This uses deterministic seeded sampling over eligible ConceptNet triples. The
same seed and coverage percentage will reproduce the same exported background
files. When `CONCEPTNET_AUTO_REBUILD_ON_CHANGE=true`, the service will detect
config drift against `conceptnet_manifest.json` on startup and regenerate the
background files automatically before loading them.

`--max-rows` remains available as an exporter-only debugging flag, but it is
not part of the normal `.env` configuration surface.

This produces:

- `data/conceptnet/conceptnet_background.metta`
- `data/conceptnet/conceptnet_background.jsonl`
- `data/conceptnet/conceptnet_manifest.json`

To add a new parser:
1. Create `parsers/your_parser.py` implementing `SemanticParser`
2. Register it in `parsers/__init__.py`
3. Set `PARSER=your_parser` in `.env`

## Local parser code (not yet on GitHub)

For parsers still in local development, mount them as volumes rather than cloning:

```yaml
# docker-compose.yml
volumes:
  - ./local-deps/manhin-parser:/deps/manhin-parser:ro
```

On your host, symlink your working directory:
```bash
mkdir -p local-deps
ln -s /path/to/your/manhin-parser local-deps/manhin-parser
```

Changes are reflected immediately without rebuilding the image. When the parser is
published to GitHub, swap the volume mount for a `git clone` in the Dockerfile.

## Local development (without Docker)

```bash
# 1. Install Ollama and pull the embedding model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text

# 2. Install SWI-Prolog 9.x
sudo add-apt-repository ppa:swi-prolog/stable
sudo apt-get install swi-prolog

# 3. Build janus_swi from source (NEVER use pip install janus-swi)
git clone https://github.com/SWI-Prolog/packages-swipy
cd packages-swipy && pip install .
cd ..

# 4. Clone and install PeTTa + PeTTaChainer
git clone https://github.com/trueagi-io/PeTTa.git
git clone https://github.com/rTreutlein/PeTTaChainer.git

cd PeTTa
sed -i "/'janus-swi'/d" setup.py   # remove the broken pip janus-swi dep
pip install -e .
cd ..

cd PeTTaChainer && pip install -e . && cd ..

# 5. Install pln-rag deps
pip install -r requirements.txt

# 6. Configure
cp .env.example .env
# Fill in OPENAI_API_KEY
# OLLAMA_URL defaults to http://localhost:11434/api/embeddings — no change needed

# 7. Run
uvicorn api.main:app --reload
```

## Compare parser outputs

Use `compare_parsers.py` to inspect how `nl2pln`, `canonical_pln`, and `manhin`
translate the same input:

```bash
python compare_parsers.py \
  --mode query \
  --text "Is Kebede smart?" \
  --context "(Inheritance Kebede Human)" \
  --context "(Implication (Inheritance $x Human) (Inheritance $x Smart))"
```

The script prints JSON and marks parsers as unavailable if their dependencies are
not installed in the current environment.

## Dependency notes

**janus_swi must always be built from source.**
The pip wheel is compiled against a specific SWI-Prolog ABI version.
If it does not match the installed SWI-Prolog, you will get:
```
janus_swi.janus.PrologError: <exception str() failed>
```
The fix is always: `git clone https://github.com/SWI-Prolog/packages-swipy && pip install .`

**Ollama must be running on the host before starting the service.**
The container reaches it via `host.docker.internal:11434`. If Ollama is not running,
ingest and query requests will fail with a connection refused error.

## Data persistence

| Path | Contents | Backed by |
|------|----------|-----------|
| `data/atomspace/kb.metta` locally, `/app/data/atomspace/kb.metta` in Docker | PLN atoms (facts + rules) | file, loaded on startup |
| `data/faiss/` | Predicate embeddings (Manhin parser) | FAISS index files |
| Qdrant volume | NL ↔ PLN sentence mappings | Docker volume |
| `~/.ollama` | Embedding model weights | host machine |

Data survives container restarts via the `pln_data` Docker volume.
Ollama model weights live on your host and never need to be re-pulled.
