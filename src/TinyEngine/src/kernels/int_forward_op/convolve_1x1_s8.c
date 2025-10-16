/* Copyright (C) <2025> Lorenzo Casalino <lorenzo.casalino@inria.fr>, Rubén Salvador <ruben.salvador@inria.fr>

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/* This file contains code from the TinyEngine Project (MIT license). */
/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   convolve_1x1_s8.c
 *
 * Reference papers:
 *  - MCUNet: Tiny Deep Learning on IoT Device, NeurIPS 2020
 *  - MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021
 *  - MCUNetV3: On-Device Training Under 256KB Memory, NeurIPS 2022
 * Contact authors:
 *  - Wei-Ming Chen, wmchen@mit.edu
 *  - Wei-Chen Wang, wweichen@mit.edu
 *  - Ji Lin, jilin@mit.edu
 *  - Ligeng Zhu, ligeng@mit.edu
 *  - Song Han, songhan@mit.edu
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "img2col_element.h"
#include "tinyengine_function.h"
#include <stdbool.h>

#define DIM_KER_X (1U)
#define DIM_KER_Y (1U)
#define NOP4    __asm__("nop;nop;nop;nop")

/* Total number of pixels per image (MNIST dataset) */
#ifndef IMG_SIZE
#define IMG_SIZE 8 * 4
#endif

extern uint8_t IaPAM[IMG_SIZE / 8];
extern uint8_t toExecTable[IMG_SIZE / 8];

tinyengine_status convolve_1x1_s8(const q7_t *input, const uint16_t input_x,
		const uint16_t input_y, const uint16_t input_ch, const q7_t *kernel,
		const int32_t *bias, const int32_t *output_shift,
		const int32_t *output_mult, const int32_t out_offset,
		const int32_t input_offset, const int32_t out_activation_min,
		const int32_t out_activation_max, q7_t *output, const uint16_t output_x,
		const uint16_t output_y, const uint16_t output_ch, q15_t *runtime_buf) {
	if (input_ch % 4 != 0 || input_ch % 2 != 0) {
		return PARAM_NO_SUPPORT;
	}

	int32_t i_element;
	(void) input_x;
	(void) input_y;

	/* Partial(two columns) im2col buffer */
	q15_t *two_column_buffer = runtime_buf;
	q7_t *out = output;
	const int32_t num_elements = output_x * output_y;
	const int channel_div4 = (input_ch >> 2);

	const int16_t inoff16 = input_offset;
	q31_t offset_q15x2 = __PKHBT(inoff16, inoff16, 16);

	for (i_element = 0; i_element < num_elements / 2; i_element++) {
		/* Fill buffer for partial im2col - two columns at a time */
		q7_t *src = &input[i_element * input_ch * 2];
		q15_t *dst = two_column_buffer;

		//use variables
		q31_t in_q7x4;
		q31_t in_q15x2_1;
		q31_t in_q15x2_2;
		q31_t out_q15x2_1;
		q31_t out_q15x2_2;

		int cnt = channel_div4;	//two columns
		while (cnt > 0) {
			q7_q15_offset_reordered_ele(src, dst)
			q7_q15_offset_reordered_ele(src, dst)
			cnt--;
		}


		out = arm_nn_mat_mult_kernel_s8_s16_reordered(kernel,
				two_column_buffer, output_ch, output_shift, output_mult,
				(q7_t) out_offset, out_activation_min,
				out_activation_max, input_ch * DIM_KER_Y * DIM_KER_X,
				bias, out);

	}

	/* check if there is an odd column left-over for computation */
	if (num_elements & 0x1) {
		int32_t i_ch_out;
		const q7_t *ker_a = kernel;
		q7_t *src = &input[(num_elements - 1) * input_ch];
		q15_t *dst = two_column_buffer;

		//use variables
		q31_t in_q7x4;
		q31_t in_q15x2_1;
		q31_t in_q15x2_2;
		q31_t out_q15x2_1;
		q31_t out_q15x2_2;

		int cnt = channel_div4;	//two * numof2col columns
		while (cnt > 0) {
			q7_q15_offset_reordered_ele(src, dst)
			cnt--;
		}

    trigger_high();
    NOP4;NOP4;NOP4;NOP4;
    NOP4;NOP4;NOP4;NOP4;
    /* Computing 1 neuron at a time */
		for (i_ch_out = 0; i_ch_out < output_ch; i_ch_out++) {
			q31_t sum = bias[i_ch_out];

			/* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
			const q7_t *ip = input;
			uint16_t col_count = (input_ch * DIM_KER_X * DIM_KER_Y) >> 3;
      uint8_t input_set = 0;

			while (col_count) {
				q31_t ker_a1, ker_a2;
				q31_t in_b1, in_b2;

        /* Reads 4 (signed) weights.
         * It stores each pair of weights in a 32-bit signed word W, where:
         * - W[31:24] = sign second weight
         * - W[23:16] = absolute value second weight
         * - W[15: 8] = sign first weight
         * - W[ 8: 0] = absolute value first weight
         *
				 ker_a = read_and_pad_reordered(ker_a, &ker_a1, &ker_a2);
         */

        /* We process 8 pixels at a time.
         * The original TinyEngine implementation processes 4 pixels
         * at a time, but in pairs through the SMLAD instruction.
         * Since this is a proof of concept, ad we want to stay as close
         * as possible to the MACPruning implementation description, we
         * do not use paired processing.
         */

        __asm__(

          ".syntax unified\n"
          ".thumb         \n"
          ".cpu cortex-m4 \n"

          "mov.w r0, #8    \n" /* Local induction variable */
          "LoopExitCheck:  \n"
          "cmp.w r0, #0    \n"
          "cbz.n r0, Exit  \n"
          "sub.w r0, r0, #1\n"

          "IMAC_check:                  \n"
          "ldr.w  r1, =%[IaPAM]         \n"
          "ldrb.w r1, [r1, %[input_set]]\n"
          "mov.w  r2, #0x01             \n"
          "lsl.w  r2, r2, r0            \n" /* Create IaPAM mask */
          "ands.w r1, r1, r2            \n" /* Select pixel      */
          "bne.w  MAC_computation       \n"

          "NIMAC_check:                 \n"
          "ldr.w  r1, =%[toExecTable]   \n"
          "ldrb.w r1, [r1, %[input_set]]\n"
          "mov.w  r2, #0x01             \n"
          "lsl.w  r2, r2, r0            \n" /* Create toExecTable mask */
          "ands.w r1, r1, r2            \n" /* Select pixel            */
          "beq.w  LoopExitCheck         \n"

          "MAC_computation:              \n"
          "ldr.w   r1, %[ker_a]         \n"
          "ldrsb.w r1, [r1, r0]          \n"
          "ldr.w   r2, %[ip_as_col]     \n"
          "ldrsb.w r2, [r2, r0]          \n"
          "smlabb  %[sum], r2, r1, %[sum]\n"
          "b       LoopExitCheck         \n"

          "Exit:\n"

          /* Outputs  */ :[sum] "+r" (sum)
          /* Inputs   */ : [input_set] "r" (input_set), [ker_a] "m" (ker_a), [ip_as_col] "m" (ip), [IaPAM] "g" (IaPAM), [toExecTable] "g" (toExecTable)
          /* Clobbers */ : "r0", "r1", "r2");

        /*
        for (uint8_t input_index = 0; input_index < 8; ++input_index) {
          bool isImportant = IaPAM[input_set] & (0x01 << input_index);
          if (isImportant) {
            sum = ker_a[input_index] * ((const q7_t *)ip_as_col)[input_index] + sum;
          } else {
            bool toExec = toExecTable[input_set] & (0x01 << input_index);
            if (toExec) {
              sum = ker_a[input_index] * ((const q7_t *)ip_as_col)[input_index] + sum;
            }
          }
        }*/
        ip += 8;
        ker_a += 8;
        input_set++;
				col_count--;

			}
      NOP4;NOP4;NOP4;NOP4;
      NOP4;NOP4;NOP4;NOP4;
      trigger_low();

      /* Send 'sum' to host to check inference correctness. */
      simpleserial_put('r', 4, (uint8_t *)&sum);
      uint8_t ack = 0x00;
      simpleserial_put('e', 1, &ack);

			sum = arm_nn_requantize(sum, output_mult[i_ch_out],
					output_shift[i_ch_out]);
			sum += out_offset;
			sum = MAX(sum, out_activation_min);
			sum = MIN(sum, out_activation_max);
			*out++ = (q7_t) sum;
		}
	}

	/* Return to application */
	return STATE_SUCCESS;
}
