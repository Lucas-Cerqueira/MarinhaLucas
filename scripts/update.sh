#!/bin/bash 

# Marinha Workspace Update Script

# Env Variables

#if [[ "$OSTYPE" == "linux-gnu" ]]; then
#    # Ubuntu
#    export MARINHA_WORKSPACE=/home/natmourajr/Workspace/Doutorado/Marinha
#    export INPUTDATAPATH=/home/natmourajr/Workspace/Doutorado/Data/Marinha
#elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
#    export MARINHA_WORKSPACE=/Users/natmourajr/Workspace/Doutorado/Marinha
#    export INPUTDATAPATH=/Users/natmourajr/Workspace/Doutorado/Data/Marinha
#fi

DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
DIR+="/setup.sh"
source $DIR

cd $MARINHA_WORKSPACE
rm -Rrf *~

git fetch
git merge
