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

OBJS=global_device_alloc_dynamic_args global_device_alloc_offset global_host_alloc_dynamic_args global_host_alloc_offset local_dynamic_args_alt_conditional local_dynamic_args local_offset pitched_ptr shared_dynamic_args shared_extern_dynamic_args shared_extern_offset shared_offset struct_global_device_alloc_dynamic_args struct_global_device_alloc_offset struct_global_host_alloc_dynamic_args struct_global_host_alloc_offset struct_local_dynamic_args struct_local_offset struct_shared_dynamic_args struct_shared_extern_dynamic_args struct_shared_extern_offset struct_shared_offset two_array_global

all: $(OBJS)

$(OBJS):
	$(NVCC) $(NVCC_FLAGS) $@.cu -DSHARED -o $@

clean:
	rm -f $(OBJS)
