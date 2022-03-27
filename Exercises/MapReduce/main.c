#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifdef AOCL
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;

void cleanup();
#endif

#define MAX_SOURCE_SIZE (0x100000)
#define DEVICE_NAME_LEN 128
static char dev_name[DEVICE_NAME_LEN];


/* Error handling: check for any return errors when called */ 
void checkReturnError(cl_int ret, const char* errorMessage){
	
    if(ret != CL_SUCCESS){

        printf("%s\n", errorMessage);
        exit(1);    //exit failure
    }

}


int main()
{
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_device_id device_id;
    cl_uint ret_num_devices;
    cl_int ret;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;

    cl_uint num_comp_units;
    size_t global_size;
    size_t local_size;


    FILE *fp;
    char fileName[] = "./mykernel.cl";
    char *source_str;
    size_t source_size;

#ifdef __APPLE__
    /* Get Platform and Device Info */
    clGetPlatformIDs(1, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);
    // we only use platform 0, even if there are more plantforms
    // Query the available OpenCL device.
    ret = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, DEVICE_NAME_LEN, dev_name, NULL);
    printf("device name= %s\n", dev_name);
#else

#ifdef AOCL  /* Altera FPGA */
    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    // Get the OpenCL platform.
    platforms[0] = findPlatform("Intel(R) FPGA Emulation");
    if(platforms[0] == NULL) {
      printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
      return false;
    }
    // Query the available OpenCL device.
    getDevices(platforms[0], CL_DEVICE_TYPE_ALL, &ret_num_devices);
    printf("Platform: %s\n", getPlatformName(platforms[0]).c_str());
    printf("Using one out of %d device(s)\n", ret_num_devices);
    ret = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    printf("device name=  %s\n", getDeviceName(device_id).c_str());
#else
#error "unknown OpenCL SDK environment"
#endif

#endif

    /* Determine global size and local size */
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS,
      sizeof(num_comp_units), &num_comp_units, NULL);
    printf("num_comp_units=%u\n", num_comp_units);

#ifdef __APPLE__
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
              sizeof(local_size), &local_size, NULL);
#endif

#ifdef AOCL  /* local size reported Altera FPGA is incorrect */
    local_size = 16;
#endif

    printf("local_size=%lu\n", local_size);
    global_size = num_comp_units * local_size;
    printf("global_size=%lu, local_size=%lu\n", global_size, local_size);

    /* Create OpenCL context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    /* Create Command Queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  

#ifdef __APPLE__
    /* Load the source code containing the kernel*/
    fp = fopen(fileName, "r");
    if (!fp) {
      fprintf(stderr, "Failed to load kernel.\n");
      exit(1);
    }

    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    /* Create Kernel Program from the source */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
              (const size_t *)&source_size, &ret);

    checkReturnError(ret, "Failed to create program from source.");
    
#else

#ifdef AOCL  /* on FPGA we need to create kernel from binary */
   /* Create Kernel Program from the binary */
   std::string binary_file = getBoardBinaryFile("mykernel", device_id);
   printf("Using AOCX: %s\n", binary_file.c_str());
   program = createProgramFromBinary(context, binary_file.c_str(), &device_id, 1);
#else
#error "unknown OpenCL SDK environment"
#endif

#endif

    /* Build Kernel Program */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    checkReturnError(ret, "Failed to build program.");   

    /* Create OpenCL Kernel */
    kernel = clCreateKernel(program, "pi_calculation", &ret);
    checkReturnError(ret, "Failed to create kernel.");   

    size_t global_vals[2] = {2048, 1};
    size_t local_vals[2] = {2, 1};

    /* Allocate requested memory depending on global vals and the size (float) */
    float *final_pi_val = (float *) calloc(global_vals[0], sizeof(float));
 
    /* Create buffers */
    cl_mem result_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, global_vals[0]*sizeof(float), NULL, &ret);
    checkReturnError(ret, "Failed to create buffers.");
    
    /* Amount of time to iterate for a better Pi value precision */
    int iVal = 694;

    /* Create kernel argument */
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&result_buffer);
    ret |= clSetKernelArg(kernel, 1, global_size*sizeof(cl_float), NULL);
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&iVal);
    
    checkReturnError(ret, "Couldn't set a kernel argument.");
    
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_vals, local_vals, 0, NULL, NULL);
    checkReturnError(ret, "Couldn't enqueue the kernel.");

    /* Enqueue kernel */
    clEnqueueReadBuffer(command_queue, result_buffer, CL_TRUE, 0, global_vals[0]*sizeof(float), (void *)final_pi_val, 0, NULL, NULL);
    checkReturnError(ret, "Couldn't read the buffer.");
       
    /* Print out final value of Pi which is pi/4 * 4 */
    printf("Pi value = %.2f\n", final_pi_val[0] * 4);
    
    /* free resources (allocated memory)*/
    free(final_pi_val);
    
    clReleaseMemObject(result_buffer);
    clReleaseCommandQueue(command_queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);

    return 0;
}

#ifdef AOCL
// Altera OpenCL needs this callback function implemented in main.c
// Free the resources allocated during initialization
void cleanup() {

}
#endif
