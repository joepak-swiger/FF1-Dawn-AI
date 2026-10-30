param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot),
    [int]$HashBudgetMiB = 16,
    [switch]$SafeClean
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath $Root).Path.TrimEnd('\')
$ReportRoot = Join-Path $Root 'reports\storage_audit'
New-Item -ItemType Directory -Path $ReportRoot -Force | Out-Null
$RunId = (Get-Date).ToUniversalTime().ToString('yyyyMMdd_HHmmss_ffffZ')
$Output = Join-Path $ReportRoot $RunId
New-Item -ItemType Directory -Path $Output -ErrorAction Stop | Out-Null
$DriveName = ([System.IO.Path]::GetPathRoot($Root)).Substring(0, 1)
$FreeBefore = (Get-PSDrive -Name $DriveName).Free

$Tracked = @{}
foreach ($line in (& git -C $Root ls-files)) { $Tracked[$line.Replace('\', '/')] = $true }
$IgnoredPrefixes = [System.Collections.Generic.List[string]]::new()
$GitStatus = @(& git -C $Root status --porcelain=v1 --ignored=matching)
foreach ($line in $GitStatus) {
    if ($line.Length -lt 4) { continue }
    $Code = $line.Substring(0, 2)
    $Name = $line.Substring(3).Replace('\', '/').TrimEnd('/')
    if ($Code -eq '!!') { $IgnoredPrefixes.Add($Name) }
}

$Files = Get-ChildItem -LiteralPath $Root -Force -File -Recurse -ErrorAction SilentlyContinue |
    Where-Object { -not $_.Attributes.HasFlag([IO.FileAttributes]::ReparsePoint) }
$Records = foreach ($File in $Files) {
    $Relative = $File.FullName.Substring($Root.Length).TrimStart('\').Replace('\', '/')
    $Top = ($Relative -split '/', 2)[0]
    if ($Relative.StartsWith('.git/')) { $Git = 'git_metadata' }
    elseif ($Tracked.ContainsKey($Relative)) { $Git = 'tracked' }
    elseif ($IgnoredPrefixes | Where-Object { $Relative -eq $_ -or $Relative.StartsWith("$_/") }) { $Git = 'ignored' }
    else { $Git = 'untracked' }
    [pscustomobject]@{
        path = $Relative; bytes = [long]$File.Length; mtime_utc = $File.LastWriteTimeUtc.ToString('o')
        git_status = $Git; top_level = $Top; extension = $(if ($File.Extension) { $File.Extension.ToLowerInvariant() } else { '[no extension]' })
        absolute_path = $File.FullName
    }
}
$Records | Select-Object path,bytes,mtime_utc,git_status | Export-Csv -LiteralPath (Join-Path $Output 'inventory.csv') -NoTypeInformation -Encoding UTF8

$CacheDirs = @(Get-ChildItem -LiteralPath $Root -Force -Directory -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -in @('__pycache__', '.pytest_cache') -and
        -not $_.FullName.Substring($Root.Length).TrimStart('\').StartsWith('.venv\') -and
        -not $_.FullName.Substring($Root.Length).TrimStart('\').StartsWith('.git\') })
$Proposals = foreach ($Dir in $CacheDirs) {
    $Relative = $Dir.FullName.Substring($Root.Length).TrimStart('\').Replace('\', '/')
    $CacheFiles = @(Get-ChildItem -LiteralPath $Dir.FullName -File -Force -Recurse -ErrorAction SilentlyContinue)
    [pscustomobject]@{
        path = $Relative; bytes = [long](($CacheFiles | Measure-Object Length -Sum).Sum)
        files = $CacheFiles.Count; purpose = 'regenerable Python bytecode/test cache'
        regeneration = 'Python imports or pytest recreate this cache'
        safe_action = 'remove only with -SafeClean; otherwise retain'
        rollback = 'cache regenerates automatically'
    }
}
$Proposals | Export-Csv -LiteralPath (Join-Path $Output 'proposed_actions.csv') -NoTypeInformation -Encoding UTF8

$HashBudget = [long]$HashBudgetMiB * 1MB
$HashBytes = [long]0
$HashGroupsSkipped = 0
$DuplicateRows = [System.Collections.Generic.List[object]]::new()
$Candidates = $Files | Group-Object Length | Where-Object { $_.Count -gt 1 -and [long]$_.Name -gt 0 } | Sort-Object { [long]$_.Name }
$HashMap = @{}
foreach ($Group in $Candidates) {
    $Cost = [long]$Group.Name * $Group.Count
    if ($HashBytes + $Cost -gt $HashBudget) { $HashGroupsSkipped++; continue }
    $HashBytes += $Cost
    foreach ($File in $Group.Group) {
        $Hash = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash
        if (-not $HashMap.ContainsKey($Hash)) { $HashMap[$Hash] = [System.Collections.Generic.List[string]]::new() }
        $HashMap[$Hash].Add($File.FullName.Substring($Root.Length).TrimStart('\').Replace('\', '/'))
    }
}
foreach ($Hash in $HashMap.Keys) {
    if ($HashMap[$Hash].Count -gt 1) {
        $RelativePaths = $HashMap[$Hash].ToArray()
        $DuplicateRows.Add([pscustomobject]@{ sha256 = $Hash; bytes_each = (Get-Item -LiteralPath (Join-Path $Root ($RelativePaths[0].Replace('/', '\')))).Length; copies = $RelativePaths.Count; paths = ($RelativePaths -join ';') })
    }
}

$BeforeBytes = [long](($Records | Measure-Object bytes -Sum).Sum)
$BeforeFiles = $Records.Count
$TopLevel = @($Records | Group-Object top_level | ForEach-Object {
    [pscustomobject]@{ name = $_.Name; files = $_.Count; bytes = [long](($_.Group | Measure-Object bytes -Sum).Sum) }
} | Sort-Object bytes -Descending)
$GitBreakdown = @($Records | Group-Object git_status | ForEach-Object {
    [pscustomobject]@{ status = $_.Name; files = $_.Count; bytes = [long](($_.Group | Measure-Object bytes -Sum).Sum) }
})
$TopFiles = @($Records | Sort-Object bytes -Descending | Select-Object -First 30 | ForEach-Object {
    [pscustomobject]@{ path = $_.path; bytes = $_.bytes; git_status = $_.git_status; mtime_utc = $_.mtime_utc }
})
$SuffixCounts = @($Records | Group-Object extension | Sort-Object Count -Descending | ForEach-Object {
    [pscustomobject]@{ extension = $_.Name; files = $_.Count }
})
$GitBytes = [long](($Files | Where-Object { $_.FullName.StartsWith((Join-Path $Root '.git') + '\') } | Measure-Object Length -Sum).Sum)
$RemovedBytes = [long]0
$RemovedFiles = 0
if ($SafeClean) {
    foreach ($Proposal in $Proposals) {
        $Target = Join-Path $Root ($Proposal.path.Replace('/', '\'))
        $Resolved = (Resolve-Path -LiteralPath $Target).Path
        if ($Resolved.StartsWith($Root + '\') -and (Split-Path -Leaf $Resolved) -in @('__pycache__', '.pytest_cache')) {
            $RemovedBytes += $Proposal.bytes; $RemovedFiles += $Proposal.files
            Remove-Item -LiteralPath $Resolved -Recurse -Force
        }
    }
}
$FreeAfter = (Get-PSDrive -Name $DriveName).Free
$Summary = [ordered]@{
    root = $Root; generated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    file_count_before_cleanup = $BeforeFiles; bytes_before_cleanup = $BeforeBytes
    bytes_after_cleanup_estimated = $BeforeBytes - $RemovedBytes
    safe_cache_files_removed = $RemovedFiles; safe_cache_bytes_removed = $RemovedBytes
    disk_free_before_bytes = $FreeBefore; disk_free_after_bytes = $FreeAfter
    git_object_store_bytes = $GitBytes; git_status_breakdown = $GitBreakdown
    top_level = $TopLevel; suffix_counts = $SuffixCounts; top_30_largest_files = $TopFiles
    duplicate_hash_bytes_read = $HashBytes; duplicate_groups = $DuplicateRows
    duplicate_size_groups_skipped_by_budget = $HashGroupsSkipped; hash_budget_bytes = $HashBudget
    cleanup_mode = $(if ($SafeClean) { 'safe-clean' } else { 'audit-only' }); cleanup_proposals = $Proposals
}
$Summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Output 'summary.json') -Encoding UTF8
$Lines = [System.Collections.Generic.List[string]]::new()
$Lines.Add('FF1-Dawn-AI local storage audit'); $Lines.Add("Root: $Root")
$Lines.Add("Files: $BeforeFiles"); $Lines.Add("Bytes before cleanup: $BeforeBytes")
$Lines.Add("Git object store: $GitBytes bytes"); $Lines.Add("Safe cache removed: $RemovedBytes bytes ($RemovedFiles files)")
$Lines.Add("Disk free: $FreeBefore -> $FreeAfter bytes"); $Lines.Add(''); $Lines.Add('Largest top-level consumers:')
foreach ($Item in ($TopLevel | Select-Object -First 20)) { $Lines.Add("- $($Item.name): $($Item.bytes) bytes ($($Item.files) files)") }
$Lines.Add(''); $Lines.Add('Largest individual files:')
foreach ($Item in $TopFiles) { $Lines.Add("- $($Item.bytes) bytes: $($Item.path) [$($Item.git_status)]") }
$Lines.Add(''); $Lines.Add("Exact duplicate groups found within hash budget: $($DuplicateRows.Count)")
$Lines.Add('No experiment, model, ROM, save state, dataset, or recording was removed.')
$Lines | Set-Content -LiteralPath (Join-Path $Output 'run_report.txt') -Encoding UTF8
Write-Output $Output
