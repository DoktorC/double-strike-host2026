/* Automatically generated source file */
#include <float.h>
#include <tinyengine_function.h>
#include <tinyengine_function_fp.h>

#include "genNN.h"
#include "genModel.h"
#include "genInclude.h"

/* Variables used by all ops */
ADD_params add_params;
int i;
int8_t *int8ptr,*int8ptr2;
int32_t *int32ptr;
float *fptr,*fptr2,*fptr3;

signed char* getInput() {
    return &buffer0[0];
}
signed char* getOutput() {
    return NNoutput;
}
void end2endinference(q7_t* img){
    invoke(NULL);
}
void invoke(float* labels){
/* layer 0:CONV_2D */
convolve_1x1_s8_oddch(&buffer0[0],1,1,32,(const q7_t*) weight0,bias0,shift0,multiplier0,-128,128,-128,127,&buffer0[32],1,1,5,sbuf);
/* layer 1:CONV_2D */
// convolve_1x1_s8(&buffer0[32],1,1,5,(const q7_t*) weight1,bias1,shift1,multiplier1,0,128,-128,127,&buffer0[0],1,1,2,sbuf);
/* layer 2:CONV_2D */
// convolve_1x1_s8_oddch(&buffer0[0],1,1,2,(const q7_t*) weight2,bias2,shift2,multiplier2,-128,0,-128,127,&buffer0[4],1,1,1,sbuf);
}
void invoke_inf(){
/* layer 0:CONV_2D */
convolve_1x1_s8_oddch(&buffer0[0],1,1,32,(const q7_t*) weight0,bias0,shift0,multiplier0,-128,128,-128,127,&buffer0[32],1,1,5,sbuf);
/* layer 1:CONV_2D */
convolve_1x1_s8(&buffer0[32],1,1,5,(const q7_t*) weight1,bias1,shift1,multiplier1,0,128,-128,127,&buffer0[0],1,1,2,sbuf);
}
