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

#include "genNN.h"
#include "hal.h"
#include "simpleserial.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if SS_VER != SS_VER_2_1
#error "Firmware supports only SimpleSerial v.2.1."
#endif

#define NOP4    __asm__("nop;nop;nop;nop")

/* The number of neurons making the MLP */
#define NUM_NEURONS 5

/* Total number of pixels per image */
#ifndef IMG_SIZE
#define IMG_SIZE 8 * 4
#endif

#ifndef PEAK_MEM
#define PEAK_MEM 168
#endif

/* WARNING:
 * the CWLITEARM's UART buffer cannot receive more than 249 bytes in a single shot,
 * the scope buffer does not capture the full inference.
 */
#define INPUT_SIZE IMG_SIZE

uint8_t IaPAM[IMG_SIZE / 8] = { 0 };

uint8_t toExecTable[(IMG_SIZE / 8) * NUM_NEURONS] = { 0 };
uint8_t ack = 0x00;

uint8_t hello(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *data) {
  char msg[] = ">>> CWLITEARM: ready to capture!";
  uint8_t msgLen = strlen(msg);

  // Between 'read' messages, the simpleserial protocol checks
  // whether the board has sent an ack indicating no error.
  // This operation can be turned off on the host side:
  //  https://chipwhisperer.readthedocs.io/en/latest/target-api.html#chipwhisperer.targets.SimpleSerial.read
  //
  // To comply with the specification, interleave the writes on the serial
  // wire by ack messages.
  simpleserial_put('r', 1, &msgLen);
  simpleserial_put('e', 1, &ack);
  simpleserial_put('r', msgLen, (uint8_t *)&msg);

  return 0x00;
}

uint8_t loadInput(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *data) {
  signed char *input = getInput();

  /* Rewrite input to 0x00 as we set it to 0xFF at startup.
   * PEAK_MEM defined in 'TinyEngine/codegen/Include/genModel.h'
   */
  memset(input, 0x00, PEAK_MEM);
  memcpy(input, data, INPUT_SIZE);

  return 0x00;
}

uint8_t loadIaPAM(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *data) {
  memcpy(IaPAM, data, len);
  return 0x00;
}

uint8_t loadToExecTable(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *data) {
  memcpy(toExecTable, data, len);
  return 0x00;
}

uint8_t infer(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *data) {
  float labels[10] = { 0 };

  invoke((float *)&labels);

  return 0x00;
}

int main(void) {

  platform_init();
  init_uart();
  trigger_setup();

  simpleserial_init();

  simpleserial_addcmd('a', IMG_SIZE, loadInput);
  simpleserial_addcmd('c', IMG_SIZE / 8, loadIaPAM);
  simpleserial_addcmd('t', (IMG_SIZE / 8) * NUM_NEURONS, loadToExecTable);
  simpleserial_addcmd('i', 0, infer);
  simpleserial_addcmd('h', 0, hello);

  signed char *input = getInput();

  while (true) {
    simpleserial_get();
  }
}
