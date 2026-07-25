@echo off
REM ============================================
REM CADRE Full Training Pipeline (Windows)
REM ============================================
REM This .bat file now just calls the resumable Python
REM orchestrator (scripts\run_pipeline.py). All the stage
REM commands and checkpointing logic live there.
REM
REM Usage:
REM   scripts\run_pipeline.bat            -> run / resume the full pipeline
REM   scripts\run_pipeline.bat --status   -> show progress, run nothing
REM   scripts\run_pipeline.bat --reset    -> forget progress, start over
REM   scripts\run_pipeline.bat --redo domain_sg  -> force one stage to re-run
REM
REM RESUMING AFTER A STOP:
REM   If you stop this (Ctrl+C, closing the window, a crash, a power cut)
REM   after, say, domain_us finishes, just run this same .bat file again.
REM   It will SKIP domain_us (and any other already-finished stage) and
REM   continue from wherever it left off - it will NOT start from stage 1.
REM   Progress is tracked in checkpoints\pipeline_state.json.
REM ============================================

echo ============================================
echo   CADRE - Full Training Pipeline (Windows)
echo ============================================
echo.

python scripts\run_pipeline.py %*
if errorlevel 1 goto :error

goto :end

:error
echo.
echo ============================================
echo   ERROR: Pipeline failed! Check output above.
echo   Progress up to the failed stage is saved.
echo   Fix the issue, then run this .bat file again
echo   to resume from the failed stage - it will NOT
echo   redo stages that already completed.
echo ============================================
exit /b 1

:end