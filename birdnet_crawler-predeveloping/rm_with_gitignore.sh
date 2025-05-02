#!/bin/bash

git ls-files -i -c -X .gitignore
echo 
read -r -p "Remove all? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    git rm --cached $(git ls-files -i -c -X .gitignore)
    echo
    echo "You should finish this task with"
    echo '   git add . && git commit -m "Removing all files in .gitignore"'
else
    echo "Nothing removed."
fi
