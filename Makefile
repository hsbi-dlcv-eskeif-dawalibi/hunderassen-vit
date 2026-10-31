# Common tasks. Run `make help` for an overview.
#
#   make setup        # once: environment, packages, dataset
#   make all          # EVERYTHING of milestone 1: data, runs, reports, why, explain, summary (resumable)
#   make smoke        # the same pipeline in minutes on a tiny subset (checks that everything runs)
#   make station1     # station 1 (adaptation): runs, augmentation, cost, report, paper
#
# RunPod: bash scripts/runpod.sh (see docs/RUNPOD.md). Finished runs are always skipped.

# Prefer the project's virtual environment (created by `make setup`), else python / python3.
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,$(shell command -v python >/dev/null 2>&1 && echo python || echo python3))
SEEDS     ?= 1 2 3
RUNS      ?= runs
RESULTS   ?= results
# Extra key=value overrides for every training run (used by `make smoke`).
OVERRIDES ?=
WHY_ARGS  ?=
EXPLAIN_ARGS ?=
export PYTHON RUNS OVERRIDES

TRAIN = bash scripts/train_if_missing.sh
# LaTeX: latexmk if installed, otherwise tectonic (small, downloads packages on demand: brew install tectonic)
LATEX = $(if $(shell command -v latexmk 2>/dev/null),latexmk -pdf -interaction=nonstopmode main.tex,tectonic -X compile main.tex)
# Apple GPUs (MPS) hit a PyTorch backward bug with ViT fine-tuning: the smoke test trains on the CPU there.
SMOKE_DEVICE = $(if $(filter Darwin,$(shell uname -s)),training.extra.use_cpu=true,)
SMOKE_EXPLAIN_DEVICE = $(if $(filter Darwin,$(shell uname -s)),--device cpu,)

# ---------------------------------------------------------------- station 1
MS1_VIT = vit_b16_head vit_b16_lora_r4 vit_b16_lora_r8 vit_b16_lora_r16 vit_b16_lora_r8_all vit_b16
MS1 = $(MS1_VIT) resnet50 convnext_t
# Augmentation comparison on one LoRA config ("light" is its main run).
AUG_CONFIG = vit_b16_lora_r8
# Optional analyses outside the station tasks (configs/extra).
EXTRA = pos_embedding_none pos_embedding_shuffled conv_stem swin_t

.PHONY: help setup all smoke station1 install data data-report debug \
	ms1 ms1-vit ms1-augment ms1-cnn-lr ms1-benchmark report-ms1 best-models why explain extra cue-conflict analysis summary pack paper tag \
	test lint format clean

help:            ## Show this help
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

# ================================================================ setup
install:         ## Install the package in editable mode with dev tools
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

setup:           ## One-time setup: .venv, PyTorch (GPU if available), project, dataset, smoke test
	bash scripts/setup.sh

data:            ## Download Oxford-IIIT Pet (~800 MB) into data/oxford_pets
	$(PYTHON) scripts/prepare_data.py

data-report:     ## Data quality of the 25 dog breeds -> results/data_quality/README.md
	$(PYTHON) scripts/data_report.py --out $(RESULTS)/data_quality

debug:           ## Smoke-test the training pipeline on a tiny subset
	$(PYTHON) scripts/train.py configs/debug.yaml

# ================================================================ everything
all:             ## EVERYTHING of milestone 1: data, station 1, why, explain, summary, results bundle (resumable)
	$(MAKE) data data-report
	$(MAKE) ms1 ms1-augment ms1-benchmark report-ms1
	$(MAKE) best-models why explain
	$(MAKE) summary pack

smoke:           ## Whole pipeline on a tiny subset, 1 seed, 1 epoch (minutes; outputs in runs_smoke/)
	$(MAKE) all RUNS=runs_smoke RESULTS=runs_smoke/results SEEDS=1 \
		OVERRIDES="data.max_samples=48 training.epochs=1 training.batch_size=8 training.fp16=false $(SMOKE_DEVICE)" \
		WHY_ARGS="--max-samples 48" EXPLAIN_ARGS="--n 16 --n-val 16 --pairs 1 --batch-size 8 $(SMOKE_EXPLAIN_DEVICE)"

# ================================================================ why so high? (beyond the station tasks)
why:             ## Why > 90 %? From-scratch runs, ImageNet heads w/o training, near duplicates, few-shot, errors
	@for m in vit_b16 resnet50; do for s in $(SEEDS); do \
		$(TRAIN) why_$${m}_scratch_s$$s configs/why/$${m}_scratch.yaml training.seed=$$s || exit 1; done; done
	$(PYTHON) scripts/why_analysis.py --out $(RESULTS)/why $(WHY_ARGS)
	$(PYTHON) scripts/report_why.py --runs-dir $(RUNS) --out $(RESULTS)/why

explain:         ## Why this breed and not the other? Grad-CAM, occlusion, contrastive maps, head focus, deletion test
	@$(PYTHON) -c "import pytorch_grad_cam, captum" 2>/dev/null || $(PYTHON) -m pip install -q "grad-cam>=1.5" "captum>=0.7"
	$(PYTHON) scripts/explain.py --runs-dir $(RUNS) --out $(RESULTS)/explain $(EXPLAIN_ARGS)
	$(PYTHON) scripts/report_explain.py --out $(RESULTS)/explain

# ================================================================ station 1 "Adaptation"
station1:        ## Station 1: runs, augmentation, cost, report, summary, paper (resumable)
	$(MAKE) data data-report ms1 ms1-augment ms1-benchmark report-ms1 summary paper

ms1:             ## Station 1: ViT (linear probing, LoRA r4/r8/r16/all, full) + ResNet-50 + ConvNeXt-T x SEEDS
	@for m in $(MS1); do for s in $(SEEDS); do \
		$(TRAIN) ms1_$${m}_s$$s configs/ms1/$$m.yaml training.seed=$$s || exit 1; done; done

ms1-vit:         ## Station 1: only the ViT adaptation methods x SEEDS
	@for m in $(MS1_VIT); do for s in $(SEEDS); do \
		$(TRAIN) ms1_$${m}_s$$s configs/ms1/$$m.yaml training.seed=$$s || exit 1; done; done

ms1-augment:     ## Station 1: augmentation none vs. strong on LoRA r=8 (light = main run) x SEEDS
	@for a in none strong; do for s in $(SEEDS); do \
		$(TRAIN) ms1_$(AUG_CONFIG)_aug_$${a}_s$$s configs/ms1/$(AUG_CONFIG).yaml \
			training.seed=$$s data.augmentation=$$a || exit 1; done; done

CNN      ?= resnet50
CNN_LRS  ?= 2e-4 5e-4 1e-3
ms1-cnn-lr:      ## Station 1 check: learning rate of a CNN on validation (CNN=resnet50|convnext_t, CNN_LRS x SEEDS, ~10 min)
	@for lr in $(CNN_LRS); do for s in $(SEEDS); do \
		$(TRAIN) lr_$(CNN)_lr$${lr}_s$$s configs/ms1/$(CNN).yaml \
			training.seed=$$s training.learning_rate=$$lr || exit 1; done; done
	$(PYTHON) scripts/lr_sweep.py $(RUNS)/lr_$(CNN)_lr* --baseline $(RUNS)/ms1_$(CNN)_s* \
		| tee $(RESULTS)/ms1/lr_sweep_$(CNN).md

ms1-benchmark:   ## Station 1: parameters, GFLOPs and latency of every method (seed 1)
	$(PYTHON) scripts/benchmark.py $$(ls -d $(RUNS)/ms1_*_s1 | grep -v _aug_)

report-ms1:      ## Station 1 report: results/ms1/REPORT.md (+ tables.tex, figures)
	$(PYTHON) scripts/report_ms1.py $(RUNS)/ms1_*_s* --out $(RESULTS)/ms1

# ================================================================ station 2 "Compression"
# ================================================================ station 3 "Generalization"
# ================================================================ optional
extra:           ## Optional analyses outside the station tasks (configs/extra) x SEEDS
	@for m in $(EXTRA); do for s in $(SEEDS); do \
		$(TRAIN) extra_$${m}_s$$s configs/extra/$$m.yaml training.seed=$$s || exit 1; done; done

cue-conflict:    ## Optional: shape-vs-texture images (style transfer, once)
	$(PYTHON) scripts/make_cue_conflict.py --per-class 10 --species dogs

analysis:        ## Optional: variants, shape bias, saliency and cost for every run
	$(PYTHON) scripts/evaluate_variants.py $(RUNS)/ms*_s*
	$(PYTHON) scripts/evaluate_shape_bias.py $(RUNS)/ms*_s*
	$(PYTHON) scripts/saliency_maps.py $(RUNS)/ms*_s* --method gradcam rollout
	$(PYTHON) scripts/benchmark.py $(RUNS)/ms*_s1

# ================================================================ results
best-models:     ## Pick the best station-1 ViT and CNN by validation accuracy -> results/ms1/best_models.json
	$(PYTHON) scripts/select_teacher.py $(RUNS) --out $(RESULTS)/ms1

summary:         ## Aggregate all runs into results/summary.{csv,md}
	$(PYTHON) scripts/summarize.py $(RUNS)/ms* --out $(RESULTS)

pack:            ## Bundle all results without model weights -> results_bundle_<date>.tar.gz
	bash scripts/pack_results.sh $(RUNS)

paper:           ## Build the station 1 paper (CVPR 2026 template) -> paper/ms1/main.pdf
	cd paper/ms1 && $(LATEX)

tag:             ## Tag the submitted state, e.g. make tag MS=ms1 (git link in the paper), then git push --tags
	git tag -a $(MS) -m "Submission $(MS)" && echo "tagged $(MS) - publish with: git push origin $(MS)"

# ================================================================ development
test:            ## Run the unit and pipeline tests (CPU)
	$(PYTHON) -m pytest

lint:            ## Static checks
	ruff check src scripts tests
	ruff format --check src scripts tests

format:          ## Auto-format the code
	ruff check --fix src scripts tests
	ruff format src scripts tests

clean:           ## Remove caches (keeps runs/ and data/)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
