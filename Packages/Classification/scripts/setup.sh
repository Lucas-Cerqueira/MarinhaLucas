#!/bin/bash 

# Marinha Workspace Setup Script

MY_WORKSPACE=$MARINHA_WORKSPACE/Packages/Classification
MY_OUTPUTPATHDATA=$MARINHA_WORKSPACE/Results/Classification


cd $MY_WORKSPACE

for i in $(ls -d */); do 
	if [ -d "$OUTPUTDATAPATH/Classification/${i%%/}" ]; then
		echo "$OUTPUTDATAPATH/Classification/${i%%/} exists"
	else 
		mkdir $OUTPUTDATAPATH/Classification/${i%%/}; 
		mkdir $OUTPUTDATAPATH/Classification/${i%%/}/mat; 
		mkdir $OUTPUTDATAPATH/Classification/${i%%/}/pict;
	fi
done
rm -rf $OUTPUTDATAPATH/Classification/functions
rm -rf $OUTPUTDATAPATH/Classification/scripts
