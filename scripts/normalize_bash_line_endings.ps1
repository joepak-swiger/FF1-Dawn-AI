param(
    [Parameter(Mandatory = $true)]
    [string[]] $Paths
)

$ErrorActionPreference = 'Stop'
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)

foreach ($path in $Paths) {
    $resolved = (Resolve-Path -LiteralPath $path).Path
    $bytes = [System.IO.File]::ReadAllBytes($resolved)
    $hasUtf8Bom = $bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF
    $content = [System.IO.File]::ReadAllText($resolved)
    $normalized = $content.Replace("`r`n", "`n").Replace("`r", "`n")
    if ($hasUtf8Bom -or $normalized -cne $content) {
        [System.IO.File]::WriteAllText($resolved, $normalized, $utf8NoBom)
        Write-Output "Normalized Bash script to UTF-8 without BOM and LF: $resolved"
    }
}
