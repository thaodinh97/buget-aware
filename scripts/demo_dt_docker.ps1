param([switch]$PrepareRaw)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'
$root = "runs/shard_dt/demo_$tag"
$base = "$root/base"
$fl = @('compose','-p','shard-fu-dt','-f','compose.shard-dt.yaml')
$rpc = @('compose','-p',"shard-demo-$tag",'-f','compose.shard-rpc.yaml')
$py = '.\.venv-gpu\Scripts\python.exe'
$data = 'data/processed/connected_v3_auxfit_seed42'
New-Item -ItemType Directory -Path $root | Out-Null
Write-Host "DEMO: $root"
if ($PrepareRaw) {
    $fresh = "$root/processed"
    & $py src/data/prepare.py --input dataset/train_test_network.csv --config configs/data_connected_auxfit.json --output $fresh
    if ($LASTEXITCODE) { throw 'Preprocessing failed' }
    & $py -c "import sys;from pathlib import Path;from src.data.loading import load_data;from scripts.shard.shard_v2 import data_content_sha256;assert data_content_sha256(load_data(Path(sys.argv[1])))==data_content_sha256(load_data(Path(sys.argv[2])))" $fresh $data
    if ($LASTEXITCODE) { throw 'Fresh data differs from mounted data' }
}
& $py -c "from pathlib import Path;from src.data.loading import load_data;from scripts.shard.shard_v2 import DATA_DEFAULT,DATA_SHA256,data_content_sha256; h=data_content_sha256(load_data(Path(DATA_DEFAULT)));print('Data hash:',h);assert h==DATA_SHA256"
if ($LASTEXITCODE) { throw 'Dataset verification failed' }
if ($env:SHARD_DATA_DIR) { throw 'Clear SHARD_DATA_DIR for this pinned demo' }
docker @fl run --rm shard -m scripts.shard.shard_service init --device cuda --shards 4 --clients 20 --seed 50042 --output $base
if ($LASTEXITCODE) { throw 'FL failed' }
$state = Get-Content "$base/state.json" -Raw | ConvertFrom-Json
$ref = Get-Content 'runs/shard_dt/base_20261110_200919/state.json' -Raw | ConvertFrom-Json
if ($state.config_hash -ne $ref.config_hash -or $state.step -ne 0) { throw 'Checkpoint design mismatch' }
'{"client":10}' | Set-Content "$root/request.json" -Encoding ascii
docker @fl run --rm shard -m scripts.shard.shard_service unlearn --original-run $base --forget-request "$root/request.json" --method shard_fixed --output "$root/local_fixed"
if ($LASTEXITCODE) { throw 'FU reference failed' }
$profiles = foreach ($policy in 'shard_full','shard_fixed') {
    $p = Get-Content "runs/shard_dt/rpc_pair_20261110_211922/$policy/rpc_report.json" -Raw | ConvertFrom-Json
    if ($p.policy -ne $policy -or !$p.verified.bit_exact) { throw 'Invalid calibration' }
    [pscustomobject]@{ Policy=$policy; Bytes=[long]$p.http_body; Seconds=[double]$p.round_seconds }
}
try {
    docker @rpc up -d --wait worker0 worker5 worker9 worker12
    if ($LASTEXITCODE) { throw 'Workers failed to start' }
    foreach ($budget in 1400000,2929920) {
        $dir = "$root/budget_$budget"
        New-Item -ItemType Directory -Path $dir | Out-Null
        $selected = $profiles | Where-Object { $_.Bytes -le $budget -and $_.Seconds -le 300 } | Select-Object -First 1
        $status = if ($null -eq $selected) { 'NO_FEASIBLE_PLAN' } else { 'ADMITTED' }
        [ordered]@{ budget_http_body=$budget; budget_seconds=300; checkpoint=$base; candidates=@($profiles); selected=$selected; status=$status } |
            ConvertTo-Json -Depth 6 | Set-Content "$dir/plan.json" -Encoding utf8
        Write-Host "Budget $budget => $status $($selected.Policy)"
        if ($budget -eq 1400000 -and $selected) { throw 'Unexpected admission' }
        if ($budget -eq 2929920 -and $selected.Policy -ne 'shard_fixed') { throw 'Unexpected boundary policy' }
        if (!$selected) { continue }
        docker @rpc run --rm coordinator run --base $base --policy $selected.Policy --output "$dir/run"
        if ($LASTEXITCODE) { throw 'RPC failed' }
        $r = Get-Content "$dir/run/rpc_report.json" -Raw | ConvertFrom-Json
        if (($r.clients -join ',') -ne '0,5,9,12' -or $r.calls -ne 40 -or !$r.first_round_bit_exact -or !$r.verified.bit_exact -or !$r.verified.structure_ok) { throw 'RPC parity failed' }
        & $py -c "import sys,torch;from pathlib import Path;a,b=map(Path,sys.argv[1:]);load=lambda p:torch.load(p,map_location='cpu',weights_only=True); pairs=[(load(a/f'shard_{j}.pt'),load(b/f'shard_{j}.pt')) for j in range(4)];assert all(u.keys()==v.keys() and all(torch.equal(t,v[k]) for k,t in u.items()) for u,v in pairs);print('4/4 local and RPC shard weights identical')" "$root/local_fixed" "$dir/run"
        if ($LASTEXITCODE) { throw 'Local/RPC checkpoint mismatch' }
        $violated = $r.http_body -gt $budget -or $r.round_seconds -gt 300
        [pscustomobject]@{ policy=$r.policy; clients=($r.clients -join ','); calls=$r.calls; http_body=$r.http_body; seconds=$r.round_seconds; budget_violated=$violated; macro_f1=$r.validation_macro_f1; fpr=$r.validation_fpr; bit_exact=$r.verified.bit_exact } | Format-List
        if ($violated) { throw 'Actual cost exceeded budget' }
    }
} finally {
    docker @rpc down
}
Write-Host "DEMO COMPLETE: $root"
