#!/bin/bash 

# Marinha Workspace Setup Script

MY_WORKSPACE=$MARINHA_WORKSPACE/Packages/NoveltyDetection
MY_OUTPUTPATHDATA=$MARINHA_WORKSPACE/Results/NoveltyDetection

cd $MY_WORKSPACE


for i in $(ls -d */); do 
	if [ -d "$OUTPUTDATAPATH/NoveltyDetection/${i%%/}" ]; then
		rm -rf $OUTPUTDATAPATH/NoveltyDetection/${i%%/}
	fi
	mkdir $OUTPUTDATAPATH/NoveltyDetection/${i%%/}; 
	mkdir $OUTPUTDATAPATH/NoveltyDetection/${i%%/}/mat; 
	mkdir $OUTPUTDATAPATH/NoveltyDetection/${i%%/}/pict; 
done
rm -rf $OUTPUTDATAPATH/NoveltyDetection/functions
rm -rf $OUTPUTDATAPATH/NoveltyDetection/scripts

