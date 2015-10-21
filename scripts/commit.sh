#!/bin/bash 

# Marinha Workspace Commit Script

# Env Variables

#if [[ "$OSTYPE" == "linux-gnu" ]]; then
    # Ubuntu
#    export MARINHA_WORKSPACE=/home/natmourajr/Workspace/Doutorado/Marinha
#elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
#    export MARINHA_WORKSPACE=/Users/natmourajr/Workspace/Doutorado/Marinha
#fi

DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
DIR+="/setup.sh"
source $DIR


cd $MARINHA_WORKSPACE
rm -Rrf *~

git add .

read -e -p "Commit Comment: " comment
git commit -m "$comment"
git push origin master


