__kernel void pi_calculation(__global float* result_buffer, __local float* local_result, int iVal) {
    
    int global_id = get_global_id(0);
    int index = iVal;
	
    /* Initialize variables */
    float pi_sum = 0.0f;
    float add_val = 1.0f;
    float temp_index;
   
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Calculate value of pi/4 */
    for(int i = 0; i < index; i++){

       temp_index = i;
        
       /* if the iteration mod 2 remainder is 0 then add the fraction */
       if(i % 2 == 0){
        
           pi_sum += (1.0f / (temp_index * 2 + add_val));

       }
       
       /* Else subtract the fraction */
       else{

           pi_sum -= (1.0f / (temp_index * 2 + add_val));

       }

    }
    
    result_buffer[global_id] = pi_sum;
   
    barrier(CLK_GLOBAL_MEM_FENCE);

}
