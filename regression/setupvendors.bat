@echo off

set PATH=%PATH%;c:\MinGW\bin;C:\Program Files\CMake\bin;C:\Program Files\doxygen\bin
rem c:\MinGW\msys\1.0\bin;

set VENDOR_DIR=e:\work\vendors
set GSL_ROOT_DIR=%VENDOR_DIR%\gsl-1.16
set PATH=%PATH%;%GSL_ROOT_DIR%\lib\Release

set LAPACK_LIB_DIR=%VENDOR_DIR%\lapack-3.4.2\lib
set LAPACK_INC_DIR=%VENDOR_DIR%\lapack-3.4.2\include

set METIS_ROOT_DIR=%VENDOR_DIR%\metis-5.1.0
set PARMETIS_ROOT_DIR=%VENDOR_DIR%\parmetis-4.0.3

set QTDIR=C:\Qt\5.3\msvc2013

set SVN_SSH=c:\\Program Files\\TortoiseSVN\\bin\\TortoisePlink.exe

rem set CAFS_Fortran_COMPILER=c:\MinGW\bin\gfortran.exe
rem set PATH=e:\work\t\cmake\bin;%PATH%

set PATH=c:\Program Files\Microsoft MPI\Bin;%PATH%
rem CMake won't work if this is added to PATH because it also provides sh.exe.
rem set PATH=c:\Program Files\Git\bin;%PATH%

set CWD=%CD%
echo cmake -DCMAKE_INSTALL_PREFIX=%CWD%..\t -G "Visual Studio 12 2013" e:\work\draco.git
