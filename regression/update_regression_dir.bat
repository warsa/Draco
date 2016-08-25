@echo off
set LOGDIR=e:\regress\logs
set GIT="c:\\Program Files\\Git\\bin\\git.exe"
echo cd e:\regress\draco
cd /d e:\regress\draco
echo %GIT% pull
%GIT% pull
echo cd e:\regress\jayenne
cd /d e:\regress\jayenne
echo %GIT% pull
%GIT% pull

rem set SVN_SSH="c:\\Program Files\\TortoiseSVN\\bin\\TortoisePlink.exe"
rem set SVN="c:\\Program Files\\TortoiseSVN\\bin\\svn.exe"
rem %SVN% update > %LOGDIR%\update_regression_dir.log

rem 2&>1
