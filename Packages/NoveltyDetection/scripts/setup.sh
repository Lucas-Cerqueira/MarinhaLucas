#!/bin/bash 

# Marinha Workspace Setup Script

source /home/lucas/Documents/IC/MarinhaLucas/scripts/setup.sh

MY_WORKSPACE=$MARINHA_WORKSPACE/Packages/NoveltyDetection
MY_OUTPUTPATHDATA=$MARINHA_WORKSPACE/Results/NoveltyDetection


cd $MY_WORKSPACE

for i in $(ls -d */); do 
	if [ -d "$OUTPUTDATAPATH/NoveltyDetection/${i%%/}" ]; then
		echo "$OUTPUTDATAPATH/NoveltyDetection/${i%%/} exists"
	else 
		mkdir $OUTPUTDATAPATH/NoveltyDetection/${i%%/}; 
		mkdir $OUTPUTDATAPATH/NoveltyDetection/${i%%/}/mat; 
		mkdir $OUTPUTDATAPATH/NoveltyDetection/${i%%/}/pict;
	fi
done
rm -rf $OUTPUTDATAPATH/NoveltyDetection/functions
rm -rf $OUTPUTDATAPATH/NoveltyDetection/scripts
