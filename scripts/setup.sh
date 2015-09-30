#!/bin/bash 

# Marinha Workspace Startup Script

# Env Variables

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    # Ubuntu
    cd ../
    export MARINHA_WORKSPACE=$(pwd)
    cd scripts
    #export MARINHA_WORKSPACE=/home/natmourajr/Workspace/Doutorado/Marinha
    export INPUTDATAPATH=/home/lucas/Documents/IC/Data
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
    cd ../
    export MARINHA_WORKSPACE=$(pwd)
   # export MARINHA_WORKSPACE=/Users/natmourajr/Workspace/Doutorado/Marinha
    export INPUTDATAPATH=/Users/lucas/Documents/IC/Data
fi

export OUTPUTDATAPATH=$MARINHA_WORKSPACE/Results
