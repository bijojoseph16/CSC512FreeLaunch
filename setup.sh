#!/bin/sh 
cd ~/llvm//llvm/tools/clang
mkdir -p tools/extra/cuda-identify-t1
echo 'add_subdirectory(cuda-identify-t1)' >> tools/extra/CMakeLists.txt
touch tools/extra/cuda-identify-t1/CMakeLists.txt
echo 'set(LLVM_LINK_COMPONENTS support)\n' >> tools/extra/cuda-identify-t1/CMakeLists.txt
echo 'add_clang_executable(cuda-identify-t1' >> tools/extra/cuda-identify-t1/CMakeLists.txt
echo '\tCudaIdentifyT1.cpp' >> tools/extra/cuda-identify-t1/CMakeLists.txt
echo '\t)' >> tools/extra/cuda-identify-t1/CMakeLists.txt
echo 'target_link_libraries(cuda-identify-t1' >> tools/extra/cuda-identify-t1/CMakeLists.txt
echo '\tclangTooling' >> tools/extra/cuda-identify-t1/CMakeLists.txt
echo '\tclangBasic' >> tools/extra/cuda-identify-t1/CMakeLists.txt
echo '\tclangASTMatchers' >> tools/extra/cuda-identify-t1/CMakeLists.txt
echo '\t)' >> tools/extra/cuda-identify-t1/CMakeLists.txt



