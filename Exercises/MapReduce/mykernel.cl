__kernel void string_search( __global float* buffer,
     int iterationVal, __local float* local_result, 
     int work_item) {
    
   const int global_id = get_global_id(0);
   const int local_id = get_local_id(0);
   
   /* initialize local data */
   local_result[0] = 0;
   local_result[1] = 0;
   local_result[2] = 0;
   local_result[3] = 0;

   /* Make sure previous processing has completed */
   barrier(CLK_LOCAL_MEM_FENCE);

   int item_offset = global_id * work_item;


   /* Make sure local processing has completed */
   barrier(CLK_GLOBAL_MEM_FENCE);

   /* Perform global reduction */

}
