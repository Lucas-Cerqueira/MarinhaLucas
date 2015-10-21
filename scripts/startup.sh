#!/bin/bash 

# Marinha Workspace Startup Script

# Env Variables

#if [[ "$OSTYPE" == "linux-gnu" ]]; then
    # Ubuntu
#    export MARINHA_WORKSPACE=/home/natmourajr/Workspace/Doutorado/Marinha
#    export INPUTDATAPATH=/home/natmourajr/Workspace/Doutorado/Data/Marinha
#elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
#    export MARINHA_WORKSPACE=/Users/natmourajr/Workspace/Doutorado/Marinha
#    export INPUTDATAPATH=/Users/natmourajr/Workspace/Doutorado/Data/Marinha
#fi

#export OUTPUTDATAPATH=$MARINHA_WORKSPACE/Results


DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
DIR+="/setup.sh"
source $DIR 

# Folder Configuration
if [ -d "$OUTPUTDATAPATH" ]; then
    read -e -p "Folder $OUTPUTDATAPATH exist, Do you want to erase it? [Y,n] " yn_erase
    if [ "$yn_erase" = "Y" ]; then
        echo "creating OUTPUTDATAPATH struct"
        rm -rf $OUTPUTDATAPATH
        mkdir $OUTPUTDATAPATH
        cd $MARINHA_WORKSPACE/Packages 
        for i in $(ls -d */)
        do 
             mkdir $OUTPUTDATAPATH/${i%%/}
        done 
#        for i in $(ls -d */)                          It is an alternative way, creating all possible subfolders in Results folder
#        do
#          mkdir $OUTPUTDATAPATH/${i%%/}
#          cd $MARINHA_WORKSPACE/Packages/$i
#          for j in $(ls -d */)
#          do
#            if [ "$j" != "functions/" -a "$j" != "scripts/" ]; then
#              mkdir $OUTPUTDATAPATH/$i/${j%%}
#            fi
#          done
#        done
        cd $MARINHA_WORKSPACE
    fi
else
    echo "OUTPUTDATAPATH: $OUTPUTDATAPATH doesnt exist"
    echo "creating OUTPUTDATAPATH struct"
    rm -rf $OUTPUTDATAPATH
    mkdir $OUTPUTDATAPATH
    cd $MARINHA_WORKSPACE/Packages
    for i in $(ls -d */)
    do 
          mkdir $OUTPUTDATAPATH/${i%%/}
    done 
#   for i in $(ls -d */)                          It is an alternative way, creating all possible subfolders in Results folder
#   do
#     mkdir $OUTPUTDATAPATH/${i%%/}
#     cd $MARINHA_WORKSPACE/Packages/$i
#     for j in $(ls -d */)
#     do
#       if [ "$j" != "functions/" -a "$j" != "scripts/" ]; then
#         mkdir $OUTPUTDATAPATH/$i/${j%%}
#       fi
#     done
#   done
    cd $MARINHA_WORKSPACE
fi

if [ -d "$OUTPUTDATAPATH/DataHandler" ]; then
	rm -rf $OUTPUTDATAPATH/DataHandler
fi

# End
