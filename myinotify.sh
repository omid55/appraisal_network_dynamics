#!/bin/sh

inotifywait -q -m -e close_write src/*.py |
while read -r filename event; do
  ./test.sh         # or "./$filename"
done
