# Start, stop and health-check the local services Sarathi's MCP servers depend on.
#
#   .\scripts\mcp-services.ps1 status     # what is up, and is it answering
#   .\scripts\mcp-services.ps1 start
#   .\scripts\mcp-services.ps1 stop
#   .\scripts\mcp-services.ps1 restart
#
# Only two services are long-running -- SearxNG and Crawl4AI, both containers.
# The MCP servers themselves are stdio processes started on demand by whichever
# client is using them, so there is nothing to start or stop for those.

param(
    [Parameter(Position = 0)]
    [ValidateSet('status', 'start', 'stop', 'restart')]
    [string]$Action = 'status'
)

$ErrorActionPreference = 'Stop'

$AppData  = Join-Path $env:APPDATA 'com.sarathi.app'
$SearxDir = Join-Path $AppData 'services\searxng'
$SearxUrl = 'http://127.0.0.1:8888'
$CrawlUrl = 'http://127.0.0.1:11235'

function Test-Endpoint {
    param([string]$Url, [int]$TimeoutSec = 10)
    try {
        $r = Invoke-WebRequest -Uri $Url -TimeoutSec $TimeoutSec -UseBasicParsing
        return $r.StatusCode -ge 200 -and $r.StatusCode -lt 400
    } catch {
        return $false
    }
}

function Start-Searxng {
    $existing = docker ps -a --filter 'name=sarathi-searxng' --format '{{.Names}}'
    if ($existing) {
        docker start sarathi-searxng | Out-Null
        Write-Host 'SearxNG: started existing container'
        return
    }

    # The secret is generated once and kept out of the tracked settings file;
    # a runtime copy carries it so settings.yml stays safe to share.
    $secretPath = Join-Path $SearxDir '.secret'
    if (-not (Test-Path $secretPath)) {
        $bytes = New-Object byte[] 32
        [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        ($bytes | ForEach-Object { $_.ToString('x2') }) -join '' | Set-Content -Path $secretPath -Encoding ascii
    }
    $secret = (Get-Content $secretPath -Raw).Trim()
    (Get-Content (Join-Path $SearxDir 'settings.yml') -Raw).Replace('__SARATHI_SEARXNG_SECRET__', $secret) |
        Set-Content -Path (Join-Path $SearxDir 'settings.runtime.yml') -Encoding utf8

    docker run -d --name sarathi-searxng --restart unless-stopped `
        -p 127.0.0.1:8888:8080 `
        -v "$SearxDir\settings.runtime.yml:/etc/searxng/settings.yml:ro" `
        -e "SEARXNG_BASE_URL=$SearxUrl/" `
        searxng/searxng:latest | Out-Null
    Write-Host 'SearxNG: created and started'
}

function Start-Crawl4ai {
    $existing = docker ps -a --filter 'name=crawl4ai' --format '{{.Names}}'
    if ($existing) {
        docker start crawl4ai | Out-Null
        Write-Host 'Crawl4AI: started existing container'
    } else {
        Write-Warning 'Crawl4AI container not found. Create it with:'
        Write-Warning '  docker run -d --name crawl4ai -p 127.0.0.1:11235:11235 --shm-size=1g unclecode/crawl4ai:latest'
    }
}

switch ($Action) {
    'start' {
        Start-Searxng
        Start-Crawl4ai
        Write-Host ''
        Write-Host 'Waiting for services to answer...'
        Start-Sleep -Seconds 12
        & $PSCommandPath status
    }
    'stop' {
        docker stop sarathi-searxng crawl4ai 2>$null | Out-Null
        Write-Host 'Stopped: sarathi-searxng, crawl4ai'
    }
    'restart' {
        & $PSCommandPath stop
        Start-Sleep -Seconds 2
        & $PSCommandPath start
    }
    'status' {
        Write-Host ''
        Write-Host 'Sarathi MCP services' -ForegroundColor Cyan
        Write-Host '--------------------'

        # Reported from whether it answers, not from whether a container exists:
        # a running container that is not serving looks identical to a healthy
        # one in `docker ps`, and is the case worth catching.
        $searx = Test-Endpoint "$SearxUrl/healthz"
        if (-not $searx) { $searx = Test-Endpoint $SearxUrl }
        $crawl = Test-Endpoint "$CrawlUrl/health"

        $mark = { param($ok) if ($ok) { 'OK  ' } else { 'DOWN' } }
        Write-Host ("  {0}  SearxNG   {1}" -f (& $mark $searx), $SearxUrl)
        Write-Host ("  {0}  Crawl4AI  {1}" -f (& $mark $crawl), $CrawlUrl)

        Write-Host ''
        Write-Host 'Registry' -ForegroundColor Cyan
        Write-Host '--------'
        $registry = Join-Path $AppData 'mcp.json'
        if (Test-Path $registry) {
            $servers = (Get-Content $registry -Raw | ConvertFrom-Json).mcpServers
            $names = $servers.PSObject.Properties.Name
            Write-Host ("  {0}  ({1} servers: {2})" -f $registry, $names.Count, ($names -join ', '))
        } else {
            Write-Warning "  $registry is missing -- launched tools will get no MCP servers."
        }

        Write-Host ''
        if (-not ($searx -and $crawl)) {
            Write-Host 'Some services are down. Start them with:' -ForegroundColor Yellow
            Write-Host '  .\scripts\mcp-services.ps1 start'
            exit 1
        }
    }
}
