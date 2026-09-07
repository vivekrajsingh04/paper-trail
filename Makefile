.PHONY: help install build serve test evals diagnose demo clean check

PY      ?= .venv/bin/python
DATASET ?= starter-datasets/starter-datasets
PAGES   ?= 40
WORKERS ?= 3

help:
	@echo "make install    create .venv and install dependencies"
	@echo "make build      ingest the starter corpus into the knowledge layer"
	@echo "make serve      run the API + UI on http://127.0.0.1:8000"
	@echo "make test       unit, reasoning and API tests (no API key needed)"
	@echo "make evals      evaluation suites, including live-database invariants"
	@echo "make diagnose   surface the system's own extraction failures"
	@echo "make demo       seed a small layer from hand-specified facts (no key)"
	@echo "make check      test + evals, the pre-commit gate"
	@echo ""
	@echo "  PAGES=40      pages per document; raise it if you are not rate-limited"
	@echo "  WORKERS=3     concurrent extraction calls"

install:
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt
	@echo "done. put a Gemini key in .env (see .env.example)"

build:
	PYTHONPATH=. $(PY) -u scripts/ingest.py --workers $(WORKERS) --max-pages $(PAGES) \
		$(DATASET)/delhivery $(DATASET)/india-macroeconomy
	PYTHONPATH=. $(PY) scripts/diagnose.py --write

serve:
	PYTHONPATH=. $(PY) -m uvicorn factlayer.api:app --reload --port 8000

test:
	PYTHONPATH=. $(PY) -m pytest -q

evals:
	PYTHONPATH=. $(PY) evals/run_evals.py --with-db

diagnose:
	PYTHONPATH=. $(PY) scripts/diagnose.py

demo:
	PYTHONPATH=. $(PY) scripts/seed_demo.py --db data/demo.db
	@echo "then:  FACTLAYER_DB=data/demo.db make serve"

check: test evals

clean:
	rm -f data/factlayer.db data/demo.db
	rm -rf data/uploads
	@echo "cache and metric aliases kept; delete data/cache to force re-extraction"
