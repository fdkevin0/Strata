@echo off
rem Strata's SYCL (Intel Arc) engine for Windows, and its ready-made zip (strata-windows-x64-sycl.zip).
rem
rem   sycl\tools\build_windows.bat              build strata.exe and package dist\strata-windows-x64-sycl.zip
rem   sycl\tools\build_windows.bat tests        also build the kernel parity tests (ctest in the build folder; they need an Arc)
rem
rem Needs: Intel oneAPI Base Toolkit for Windows (the DPC++/C++ compiler icx and oneMKL; 2025.0 or newer), Visual Studio
rem 2022 (or 2019) Build Tools with the C++ workload (oneAPI's compiler uses MSVC's headers, linker and Windows SDK), CMake
rem 3.24+, Ninja, Python 3.10+ and git.  No Docker and no WSL: this is the native counterpart of sycl/tools/build.sh.  icx
rem compiles the engine in its MSVC-style mode, cl compiles ggml's C files (as llama.cpp's own Windows SYCL build).
rem
rem Settings (environment variables, all optional):
rem   AOT              the device to compile ahead of time for: bmg-g21 (B580, B570, Pro B60), bmg-g31 (Pro B70), ...
rem                    (ocloc ids lists them).  Empty: SPIR-V, compiled for the card at the first start (about a minute),
rem                    the build every Arc runs (the A-series has no AOT build, as on Linux).
rem   I8_ESIMD         AUTO (default), ON or OFF: the fused int8 ESIMD expert kernels (sycl/tools/build.sh, #3219)
rem   ONEAPI_ROOT      the oneAPI folder (default: what setvars.bat set, else %ProgramFiles(x86)%\Intel\oneAPI)
rem   BUILD_DIR        <repo>\build-sycl-win      DIST_DIR  <repo>\dist
rem   STRATA_GGML_DIR  a llama.cpp checkout at the pinned commit (default: CMake fetches it)
rem   STRATA_SYCL_NATIVE  1 = ggml-cpu for THIS PC's CPU (default: an AVX2 baseline, so the zip runs on other PCs)
setlocal EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "SRC=%%~fI"
if not defined BUILD_DIR set "BUILD_DIR=%SRC%\build-sycl-win"
if not defined DIST_DIR set "DIST_DIR=%SRC%\dist"
if not defined I8_ESIMD set "I8_ESIMD=AUTO"
set "TESTS=OFF"
if /i "%~1"=="tests" set "TESTS=ON"
set "PORTABLE=ON"
if "%STRATA_SYCL_NATIVE%"=="1" set "PORTABLE=OFF"

rem ---- 1. MSVC: the linker, the C runtime and the Windows SDK (icx uses them); before setvars, which looks for them
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS="
if exist "%VSWHERE%" for /f "delims=" %%V in ('"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do set "VS=%%V"
if not defined VS (echo Visual Studio Build Tools with the C++ workload are needed & exit /b 1)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

rem ---- 2. oneAPI: icx, oneMKL, the SYCL runtime (setvars.bat once per console; a second call is refused, so it is skipped)
if not defined SETVARS_COMPLETED (
  if not defined ONEAPI_ROOT set "ONEAPI_ROOT=%ProgramFiles(x86)%\Intel\oneAPI"
  if not exist "!ONEAPI_ROOT!\setvars.bat" (echo Intel oneAPI Base Toolkit not found in "!ONEAPI_ROOT!" - install it or set ONEAPI_ROOT & exit /b 1)
  call "!ONEAPI_ROOT!\setvars.bat" intel64 >nul || exit /b 1
)
where icx >nul 2>nul || (echo icx is not on PATH after setvars.bat: the oneAPI DPC++/C++ compiler is needed & exit /b 1)
if not defined MKLROOT (echo MKLROOT is not set after setvars.bat: oneMKL is needed & exit /b 1)

rem ---- 3. configure + build
set "GGML="
if defined STRATA_GGML_DIR set "GGML=-DSTRATA_GGML_DIR=%STRATA_GGML_DIR:\=/%"
if not exist "%BUILD_DIR%\build.ninja" (
  cmake -G Ninja -S "%SRC%\sycl" -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx "-DSTRATA_SYCL_AOT=%AOT%" "-DSTRATA_SYCL_I8_ESIMD=%I8_ESIMD%" ^
    -DSTRATA_SYCL_PARITY=%TESTS% -DSTRATA_SYCL_PORTABLE=%PORTABLE% %GGML% || exit /b 1
) else (
  rem an existing build folder keeps its cache: the switch given now reaches it (as build.sh)
  cmake -S "%SRC%\sycl" -B "%BUILD_DIR%" "-DSTRATA_SYCL_I8_ESIMD=%I8_ESIMD%" >nul || exit /b 1
)
if "%TESTS%"=="ON" (
  cmake --build "%BUILD_DIR%" || exit /b 1
) else (
  cmake --build "%BUILD_DIR%" --target strata || exit /b 1
)

rem ---- 4. the zip: strata.exe, the oneAPI runtime DLLs it loads (the SYCL runtime, its Level Zero adapter, oneMKL), licenses
python "%SRC%\sycl\tools\package_windows.py" --build "%BUILD_DIR%" --aot "%AOT%" --out "%DIST_DIR%" || exit /b 1
echo.
echo Done: %DIST_DIR%\strata-windows-x64-sycl.zip
endlocal
