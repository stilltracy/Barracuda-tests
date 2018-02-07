NTHREADS ?= 32
NBLOCKS ?= 4
SM ?= 50
# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include  -D_FORCE_INLINES -arch=sm_$(SM) --cudart=shared -DTHREADS=$(NTHREADS) -DBLOCKS=$(NBLOCKS) -rdc=true


# 'make dbg=1' enables NVCC debugging
ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0
else
	NVCC_FLAGS += -O2
endif

OBJS=device_bad_alloc device_bad_memcpy device_bad_memset device_early_dealloc_w device_uninit_memcpy device_uninit_r host_bad_alloc host_bad_memcpy host_bad_memset host_early_dealloc_w host_uninit_memcpy host_uninit_r leaky_pointers_child_kernel leaky_pointers local_uninit_r pitched_ptr_bad_memcpy shared_extern_uninit_r shared_uninit_r

all: $(OBJS)

$(OBJS):
	$(NVCC) $(NVCC_FLAGS) $@.cu -DSHARED -o $@

clean:
	rm -f $(OBJS)
