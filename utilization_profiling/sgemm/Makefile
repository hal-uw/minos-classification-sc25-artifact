all: sgemm_nvidia gen_data

sgemm_nvidia: sgemm.cu
		nvcc -arch=compute_80  -code=sm_80 -o $@ $< -lcuda -lcurand -lcudart -lcublas

gen_data: gen_data.cpp
		g++ -o $@ $<

clean:

		rm -f sgemm_nvidia gen_data