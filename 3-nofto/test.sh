#!/bin/bash
array=( $@ )
len=${#array[@]}
_outfile=${array[$len-1]}
_args=${array[@]:0:$len-1}

[ -e "$_outfile" ] && rm "$_outfile"

for var in $_args
do
    ./test_one_city.sh "$var" "$_outfile"
done
