#!/usr/bin/env bash

build_type=$1
if [ -z "$build_type" ]; then
    build_path="build"
else
    build_path="build-${build_type}"
fi


cmake_defs="-DCMAKE_BUILD_TYPE=${build_type} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
build_path="build-${build_type}"

# begin building...
rm -rf ${build_path}
mkdir ${build_path}
cd ${build_path}
cmake ${cmake_defs} ..
NPROC=`nproc`
if [ $NPROC -lt 2 ]; then
    NPROC=2
fi
make -j `expr $NPROC - 1` 2>&1 | tee make_output.log
cd ..
