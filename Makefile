.PHONY: test lint

test:
	python3 -m unittest discover -s tests -v

# A NEW TOP-LEVEL MODULE IS UNLINTED UNTIL IT IS ADDED HERE.
lint:
	ruff check lazy_eval.py make_splits.py qat_lora.py qat_smoke.py sample_train.py train.py export_gguf.py qlora_train.py tests docs/figures.py
	@echo "lint OK"
