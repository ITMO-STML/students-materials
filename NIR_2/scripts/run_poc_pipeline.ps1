# run_poc_pipeline.ps1
# PoC ablation grid - first signal run.
# n_train=100, n_test=200, single seed.
# Expected time: ~4 hours (CN production dominates).

$ErrorActionPreference = "Stop"
$startTime = Get-Date
$projectRoot = "W:\Jupyter\NIR_2"
Set-Location $projectRoot

# ----------------------------------------------------------------------
# Helper: run a pipeline step with timing and exit-code check
# ----------------------------------------------------------------------
function Step([string]$Name, [scriptblock]$Cmd) {
    Write-Host ""
    Write-Host "========== $Name ==========" -ForegroundColor Cyan
    Write-Host "Started: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
    $stepStart = Get-Date
    & $Cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name (exit $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
    $elapsed = (Get-Date) - $stepStart
    $mins = [math]::Round($elapsed.TotalMinutes, 1)
    Write-Host "DONE: $Name in $mins min" -ForegroundColor Green
}

Write-Host ""
Write-Host "  =====================================" -ForegroundColor Yellow
Write-Host "   PoC PIPELINE - first signal run"      -ForegroundColor Yellow
Write-Host "   n_train=100, n_test=200, seed=42"     -ForegroundColor Yellow
Write-Host "   ETA: ~4 hours (CN prod dominates)"    -ForegroundColor Yellow
Write-Host "  =====================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "NOTE: D/E conditions will be weak (GP coverage ~3% on new train)." -ForegroundColor DarkYellow
Write-Host "Main hypothesis C vs B is measured in full."                       -ForegroundColor DarkYellow
Write-Host ""

# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

# 1. CN production (main long stage)
Step "1/7 CN production (100 examples, ~3.5h)" {
    python scripts\05_run_cn_pipeline.py --n-examples 100
}

# 2. BM25 mining
Step "2/7 BM25 mining (100 train queries, ~2 min)" {
    python scripts\06_mine_bm25_negatives.py --n-examples 100
}

# 3. Build train data (5 ablation conditions)
Step "3/7 Build train data (n_train=100, n_test=200)" {
    python scripts\07_build_train_data.py --n-train 100 --n-test 200
}

# 4. Train all 5 conditions (single seed 42)
Step "4/7 Train all 5 conditions (~15 min)" {
    python scripts\08_train_retriever.py --conditions A B C D E
}

# 5. Eval all 5 fine-tuned
Step "5/7 Eval fine-tuned models (~15 min)" {
    python scripts\09_eval_retriever.py --conditions A B C D E
}

# 6. Eval baseline (no fine-tune)
Step "6/7 Eval baseline RoSBERTa (no FT, ~3 min)" {
    python scripts\09_eval_retriever.py --baseline
}

# 7. Compare ablations - final report
Step "7/7 Compare ablations (final report)" {
    python scripts\10_compare_ablations.py
}

# ----------------------------------------------------------------------
# Final summary
# ----------------------------------------------------------------------
$totalElapsed = (Get-Date) - $startTime
$totalMins = [math]::Round($totalElapsed.TotalMinutes, 1)
$totalHours = [math]::Round($totalElapsed.TotalHours, 2)

Write-Host ""
Write-Host "  =====================================" -ForegroundColor Yellow
Write-Host "   PIPELINE DONE"                        -ForegroundColor Yellow
Write-Host "   Total time: $totalMins min ($totalHours h)" -ForegroundColor Yellow
Write-Host "  =====================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "Main output files:" -ForegroundColor Cyan
Write-Host "  outputs\eval\comparison.md         - MAIN REPORT (start here)"
Write-Host "  outputs\eval\comparison.json       - machine-readable results"
Write-Host "  outputs\eval\comparison_chart.png  - bar chart by conditions"
Write-Host ""
Write-Host "Per-condition reports:" -ForegroundColor Cyan
Write-Host "  outputs\eval\A_seed42\report.md"
Write-Host "  outputs\eval\B_seed42\report.md"
Write-Host "  outputs\eval\C_seed42\report.md   <- main hypothesis"
Write-Host "  outputs\eval\D_seed42\report.md   (weak, GP coverage ~3%)"
Write-Host "  outputs\eval\E_seed42\report.md   (weak, GP coverage ~3%)"
Write-Host "  outputs\eval\BASE\report.md       - pre-fine-tune sanity"
Write-Host ""
Write-Host "Intermediate reports:" -ForegroundColor Cyan
Write-Host "  outputs\cn_pipeline\prod\report.md         - CN diagnostics"
Write-Host "  outputs\bm25_negatives\prod\report.md      - BM25 diagnostics"
Write-Host "  data\train\report.md                       - condition files composition"
Write-Host "  outputs\training\<cond>_seed42\report.md   - training diagnostics"
Write-Host ""
Write-Host "OPEN: outputs\eval\comparison.md -> 'PoC Decision' section" -ForegroundColor Green
