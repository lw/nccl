g++ demo.cpp -o demo -O0 -g -I /usr/local/cuda/include -I build/include/ --std=c++11 -L/usr/local/cuda/lib64 -lcudart -lrt -Lbuild/lib -ldl -lpthread -lnccl_static
