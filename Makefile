.PHONY: install lint format test test-pg up down seed demo eval

CATALOGS = -c data/examples/home_uniex.yaml -c data/examples/host_uej.yaml
OFFLINE = ERASMATCH_EMBEDDER=hashing ERASMATCH_LLM=none

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest

# Integration tests against the compose database (drops and recreates erasmatch_test).
test-pg:
	docker compose exec db createdb -U erasmatch erasmatch_test || true
	ERASMATCH_TEST_DATABASE_URL=postgresql+psycopg://erasmatch:erasmatch@localhost:5432/erasmatch_test pytest -m postgres

up:
	docker compose up -d --build

down:
	docker compose down

seed:
	docker compose exec api erasmatch import data/examples/home_uniex.yaml data/examples/host_uej.yaml

# No database, no model: offline embedder on the example catalogs.
demo:
	$(OFFLINE) erasmatch la UNIEX UEJ --courses RETI,BASIDATI,CALCOLATORI,INGSW,IA,WEB $(CATALOGS)

eval:
	$(OFFLINE) erasmatch eval data/examples/gold_uniex_uej.yaml $(CATALOGS)
