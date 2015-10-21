#!/bin/bash 

# Marinha Workspace Startup Script

# Env Variables

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    # Ubuntu
    DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
    export MARINHA_WORKSPACE=$(cd "$( dirname "$DIR" )" && pwd )
    export INPUTDATAPATH=/home/lucas/Documents/IC/Data/Marinha
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
    cd ../
    export MARINHA_WORKSPACE=$(pwd)
   # export MARINHA_WORKSPACE=/Users/natmourajr/Workspace/Doutorado/Marinha
    export INPUTDATAPATH=/Users/lucas/Documents/IC/Data/Marinha
fi

export OUTPUTDATAPATH=$MARINHA_WORKSPACE/Results
