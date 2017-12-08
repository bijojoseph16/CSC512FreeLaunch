#!/bin/sh 
tar -xvf project.tar
tar -xvf cuda-identify-T2.tar
tar -xvf cuda-identify-T3.tar
tar -jxf lonestargpu-2.0.tar.bz2
unzip cub-1.7.4.zip -d ~
cd lonestargpu-2.0
ln -s ../cub-1.7.4 cub-1.7.4
cd ~
mv cuda-identify-T2/ ~/llvm/llvm/tools/clang/tools/extra/
mv cuda-identify-T3/ ~/llvm/llvm/tools/clang/tools/extra/
cd ~/llvm//llvm/tools/clang
echo 'add_subdirectory(cuda-identify-T2)' >> tools/extra/CMakeLists.txt
echo 'add_subdirectory(cuda-identify-T3)' >> tools/extra/CMakeLists.txt


