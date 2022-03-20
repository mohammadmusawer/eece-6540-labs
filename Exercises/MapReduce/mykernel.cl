__kernel void string_search( __global float* buffer,
     int iterationVal, __local float* local_result, 
     int work_item) {
    
   const uint global_id = get_global_id(0);
   const uint local_id = get_local_id(0);
   
   /* initialize local data
   local_result[0] = 0;
   local_result[1] = 0;
   local_result[2] = 0;
   local_result[3] = 0;
   */

   /* Make sure previous processing has completed */
   barrier(CLK_LOCAL_MEM_FENCE);

   float item_offset = global_id * work_item * iterationVal;
   float piVal = 0.0; // initialize pi value

   /* Calculation by each work item */
   for(uint i = 0; i <= iterationVal; i++){
       piVal += 4.0 / (1.0 + ((float) i + item_offset));
   }
   
   /* Store pi value in the local_result array */
   local_result[global_id] = piVal;

   /* Make sure local processing has completed */
   barrier(CLK_GLOBAL_MEM_FENCE);

   /* Perform global reduction */

}
