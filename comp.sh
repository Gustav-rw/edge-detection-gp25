nvcc -c -o gpu_main.o gpu_main.cu
nvcc -c -o support.o support.cu
nvcc gpu_main.o support.o -o gpu_opt33_4
echo "done"