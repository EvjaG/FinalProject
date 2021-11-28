@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "UserPath="
for /F "skip=2 tokens=1,2*" %%G in ('%SystemRoot%\System32\reg.exe query "HKCU\Environment" /v "Path" 2^>nul') do if /I "%%G" == "Path" (
    if /I "%%H" == "REG_EXPAND_SZ" (call set "UserPath=%%I") else if /I "%%H" == "REG_SZ" set "UserPath=%%I"
    if defined UserPath goto UserPathRead
)
echo User PATH is not defined or has no string value.
goto EndBatch

:UserPathRead
setlocal EnableDelayedExpansion
echo The user PATH is: !UserPath!
endlocal

:EndBatch
echo/
pause
endlocal