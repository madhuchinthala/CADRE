@echo off
REM ============================================
REM CADRE Full Training Pipeline (Windows)
REM ============================================
REM Run all 7 parts in sequence.
REM Usage: scripts\run_pipeline.bat
REM ============================================

echo ============================================
echo   CADRE — Full Training Pipeline (Windows)
echo ============================================
echo.

REM Part 1 — Load ^& Freeze Backbone
echo [PART 1/7] Loading and freezing VLA backbone...
python -m src.models.vla_backbone --model_path checkpoints/llava-v1.5-7b --verify
if errorlevel 1 goto :error

REM Part 2-4 — Train domains with EWC + Replay (correct dataset per domain,
REM resolved automatically inside get_dataloader() from base_config.yaml)
REM 5 epochs, capped at 1500 samples/epoch (~14 min/epoch at ~2.2s/it, batch_size=4)
echo.
echo [PART 2-4] Training domain_us...
python -m src.continual.continual_trainer --config configs/base_config.yaml --domain domain_us --epochs 5 --device cuda --max-samples-per-epoch 1500
if errorlevel 1 goto :error

echo.
echo [PART 2-4] Training domain_sg...
python -m src.continual.continual_trainer --config configs/base_config.yaml --domain domain_sg --epochs 5 --device cuda --max-samples-per-epoch 1500
if errorlevel 1 goto :error

echo.
echo [PART 2-4] Training domain_eu...
python -m src.continual.continual_trainer --config configs/base_config.yaml --domain domain_eu --epochs 5 --device cuda --max-samples-per-epoch 1500
if errorlevel 1 goto :error

echo.
echo [PART 2-4] Training domain_rainy...
python -m src.continual.continual_trainer --config configs/base_config.yaml --domain domain_rainy --epochs 5 --device cuda --max-samples-per-epoch 1500
if errorlevel 1 goto :error

REM Part 5 — Train Domain Router
echo.
echo [PART 5/7] Training domain router...
python -m src.router.domain_router --config configs/router_config.yaml --domains domain_us,domain_sg,domain_eu,domain_rainy --epochs 20
if errorlevel 1 goto :error

REM Part 6 — Train Output Heads
echo.
echo [PART 6/7] Training output heads...
python -m src.heads.integration_layer --config configs/heads_config.yaml --heads waypoint,hazard,regulation,weather --domains domain_us,domain_sg,domain_eu,domain_rainy --epochs 15
if errorlevel 1 goto :error

REM Part 7 — Run CADRE-Bench
echo.
echo [PART 7/7] Running CADRE-Bench evaluation...
python -m src.benchmark.cadre_bench --config configs/benchmark_config.yaml --domains domain_us,domain_sg,domain_eu,domain_rainy --output_dir outputs/cadre_bench
if errorlevel 1 goto :error

echo.
echo ============================================
echo   Pipeline complete!
echo   Results in: outputs\cadre_bench\
echo ============================================
goto :end

:error
echo.
echo ============================================
echo   ERROR: Pipeline failed! Check output above.
echo ============================================
exit /b 1

:end