#!/bin/bash

#------------------------------------------------------------------------------#
# Generate a list of authors by using git annotate
#------------------------------------------------------------------------------#

# Should be run from the draco top-level source directory.
if ! test -f CMakeLists.txt; then
  echo "ERROR: Must be run from the top level draco source directory."
  exit 1
fi
if ! test -f README.md; then
  echo "ERROR: Must be run from the top level draco source directory."
  exit 1
fi

run () {
   echo $1
   if ! test $dry_run; then
      eval $1
   fi
}

#------------------------------------------------------------------------------#
# List of contributors
author_list_file=`mktemp`
git shortlog -s | cut -c8- &> $author_list_file

# Put the user list into an array (some entries may have spaces).
IFS=$'\r\n' GLOBIGNORE='*' command eval 'user_list=($(cat $author_list_file))'

#------------------------------------------------------------------------------#
# http://stackoverflow.com/questions/1265040/how-to-count-total-lines-changed-by-a-specific-author-in-a-git-repository

# count loc for each author and report.
author_loc=`mktemp`
for name in "${user_list[@]}"; do

  numlines=`git log --author="$name" --pretty=tformat: --numstat | awk '{ add += $1; subs += $2; loc += $1 - $2; sum += $1 + $2 } END { printf "%s:\n", loc }'`
  echo "$numlines$name"

done | sort -rn > $author_loc

cat $author_loc

# while read line; do
#   echo $line | sed -e 's/\([0-9]*\):/current_developers[\1]=/'
# done < $author_loc

#------------------------------------------------------------------------------#
# Old svn based script modified for git
# # Build a list of files to inspect.
# files=`find . -name '*.hh' -o -name '*.cc' -o -name '*.txt' \
#          -o -name '*.cmake' -o -name '*.in' -o -name '*.h'`

# # generate full annotations for these files.
# author_per_loc=`mktemp`
# for file in $files; do
#   echo "annotate $file"
#   git annotate --line-porcelain $file |  grep '^author ' | sed -e 's/author //' >> $author_per_loc
# done

# # generate a list of users identified by 'git annotate'
# #user_list=`cat $author_per_loc | awk '{print $3}' | sort -u`
# author_list=`mktemp`
# cat $author_per_loc | sort -u | grep -v Not > $author_list

# # Put the user list into an array (some entries may have spaces).
# IFS=$'\r\n' GLOBIGNORE='*' command eval 'user_list=($(cat $author_list))'

# # count loc for each author and report.
# author_loc=`mktemp`
# for name in "${user_list[@]}"; do
#   if ! test "$name" == "kgt" && ! test "$name" == "Kelly (KT) Thompson" && \
#      ! test "$name" == 107638; then
#     numlines=`grep -c "$name" $author_per_loc`
#     echo "$numlines:$name"
#   fi
# done | sort -rn > $author_loc

# while read line; do
#   echo $line | sed -e 's/\([0-9]*\):/current_developers[\1]=/'
# done < $author_loc



#------------------------------------------------------------------------------#
# end alist.sh
#------------------------------------------------------------------------------#
