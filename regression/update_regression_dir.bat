@echo off
set LOGDIR=e:\cdash\logs
set GIT="c:\\Program Files\\Git\\bin\\git.exe"
cd /d e:\cdash\draco
%GIT% pull

rem set SVN_SSH="c:\\Program Files\\TortoiseSVN\\bin\\TortoisePlink.exe"
rem set SVN="c:\\Program Files\\TortoiseSVN\\bin\\svn.exe"
rem %SVN% update > %LOGDIR%\update_regression_dir.log

rem 2&>1
