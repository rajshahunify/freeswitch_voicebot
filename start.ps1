# =============================================================================
# start.ps1 — VoiceBot Startup Script for Windows
# =============================================================================
# Usage:
#   .\start.ps1              → Start with Gemini AI (default, no Ollama)
#   .\start.ps1 gemini       → Start with Gemini AI explicitly
#   .\start.ps1 ollama       → Start with local Ollama (auto-downloads model)
#   .\start.ps1 stop         → Stop all containers
#   .\start.ps1 restart      → Restart all containers
#   .\start.ps1 logs         → Tail logs from voicebot
#   .\start.ps1 status       → Show container status
#   .\start.ps1 build        → Rebuild the Docker image (after code changes)
# =============================================================================

param(
    [string]$Mode = "gemini"
)

$ErrorActionPreference = "Stop"

function Write-Header {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "   FreeSWITCH AI VoiceBot" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
}

function Write-Step {
    param([string]$msg)
    Write-Host "  >> $msg" -ForegroundColor Yellow
}

function Write-OK {
    param([string]$msg)
    Write-Host "  [OK] $msg" -ForegroundColor Green
}

function Write-Info {
    param([string]$msg)
    Write-Host "  [i]  $msg" -ForegroundColor White
}

Write-Header

switch ($Mode.ToLower()) {

    # ── Gemini mode (default) ──────────────────────────────────────────────
    { $_ -eq "gemini" -or $_ -eq "" } {
        Write-Host "  Mode: Gemini AI (cloud LLM)" -ForegroundColor Magenta
        Write-Host ""

        # Update .env to use Gemini
        (Get-Content ".env") -replace "^LLM_PROVIDER=.*", "LLM_PROVIDER=gemini" | Set-Content ".env"
        Write-OK ".env updated: LLM_PROVIDER=gemini"

        Write-Step "Starting voicebot (Ollama not started)..."
        docker compose up -d

        Write-OK "Started!"
        Write-Info "STT:  remote (http://164.52.203.140:8890)"
        Write-Info "LLM:  Gemini Flash API"
        Write-Info "TTS:  Edge-TTS (en-US-GuyNeural)"
        Write-Info ""
        Write-Info "Call extension 5000 in Zoiper to test."
        Write-Info "Logs: .\start.ps1 logs"
    }

    # ── Ollama mode ────────────────────────────────────────────────────────
    "ollama" {
        # Read which model is configured
        $model = (Get-Content ".env" | Where-Object { $_ -match "^OLLAMA_MODEL=" }) -replace "^OLLAMA_MODEL=", ""
        if (-not $model) { $model = "qwen2.5:0.5b" }

        Write-Host "  Mode: Ollama (local LLM, model: $model)" -ForegroundColor Magenta
        Write-Host ""

        # Update .env to use Ollama
        (Get-Content ".env") -replace "^LLM_PROVIDER=.*", "LLM_PROVIDER=ollama" | Set-Content ".env"
        (Get-Content ".env") -replace "^#\s*OLLAMA_URL=http://ollama.*", "OLLAMA_URL=http://ollama:11434" | Set-Content ".env"
        Write-OK ".env updated: LLM_PROVIDER=ollama"

        Write-Step "Starting Ollama + model pull + voicebot..."
        Write-Info "(First run: model '$model' will be downloaded — this takes a few minutes)"
        Write-Info ""

        docker compose --profile ollama up -d

        Write-Host ""
        Write-OK "Started!"
        Write-Info "STT:  remote (http://164.52.203.140:8890)"
        Write-Info "LLM:  Ollama / $model (local, no cost)"
        Write-Info "TTS:  Edge-TTS (en-US-GuyNeural)"
        Write-Info ""
        Write-Info "Model download progress: .\start.ps1 logs-ollama"
        Write-Info "Call extension 5000 in Zoiper to test."
    }

    # ── Stop ───────────────────────────────────────────────────────────────
    "stop" {
        Write-Step "Stopping all containers..."
        docker compose --profile ollama down
        Write-OK "All containers stopped."
    }

    # ── Restart ────────────────────────────────────────────────────────────
    "restart" {
        Write-Step "Restarting..."
        docker compose --profile ollama restart
        Write-OK "Restarted."
    }

    # ── Build ──────────────────────────────────────────────────────────────
    "build" {
        Write-Step "Rebuilding Docker image (this takes 5-15 mins)..."
        docker compose build --no-cache
        Write-OK "Build complete. Run '.\start.ps1' to start."
    }

    # ── Logs ───────────────────────────────────────────────────────────────
    "logs" {
        Write-Step "Tailing voicebot logs (Ctrl+C to stop)..."
        docker logs -f freeswitch-voicebot
    }

    "logs-ollama" {
        Write-Step "Tailing Ollama model pull logs..."
        docker logs -f voicebot-ollama-model-pull
    }

    # ── Status ─────────────────────────────────────────────────────────────
    "status" {
        Write-Step "Container status:"
        docker compose --profile ollama ps
        Write-Host ""
        Write-Step "Health check:"
        try {
            $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 3
            Write-OK "Voicebot is healthy"
            Write-Info "STT: $($health.providers.stt)"
            Write-Info "LLM: $($health.providers.llm)"
            Write-Info "Active calls: $($health.capacity.active_calls)/$($health.capacity.max_concurrent)"
        } catch {
            Write-Host "  [!] Voicebot not responding on :8000 (may still be starting up)" -ForegroundColor Red
        }
    }

    # ── Unknown ────────────────────────────────────────────────────────────
    default {
        Write-Host "  Unknown mode: '$Mode'" -ForegroundColor Red
        Write-Host ""
        Write-Host "  Usage:" -ForegroundColor White
        Write-Host "    .\start.ps1              Start with Gemini (default)" -ForegroundColor Gray
        Write-Host "    .\start.ps1 gemini       Start with Gemini explicitly" -ForegroundColor Gray
        Write-Host "    .\start.ps1 ollama       Start with local Ollama" -ForegroundColor Gray
        Write-Host "    .\start.ps1 stop         Stop everything" -ForegroundColor Gray
        Write-Host "    .\start.ps1 restart      Restart containers" -ForegroundColor Gray
        Write-Host "    .\start.ps1 build        Rebuild Docker image" -ForegroundColor Gray
        Write-Host "    .\start.ps1 logs         Tail voicebot logs" -ForegroundColor Gray
        Write-Host "    .\start.ps1 status       Show status + health check" -ForegroundColor Gray
    }
}

Write-Host ""
