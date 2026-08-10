# hive_assist — run from this directory.
PY ?= ../.venv/bin/python

.PHONY: all test figures d1 d2 d3 d4 clean

all: test figures

test:
	$(PY) -m pytest tests/

# Each module prints its own result table and writes its figure, so `figures`
# doubles as the "show me the findings" target.
figures: d1 d2 d3 d4

d1:
	$(PY) -m hive.frames
	$(PY) -m hive.nullspace
	$(PY) -m hive.anchored_isam2

d2:
	$(PY) -m hive.mission_fsm

d3:
	$(PY) -m hive.standoff

d4:
	$(PY) -m hive.loss_model

clean:
	rm -rf figures/*.png .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
