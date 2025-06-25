#ifndef _STOREREAD_H_
#define _STOREREAD_H_

#include<stdio.h>

#include "network.h"
#include "activation.h"

void saveNetwork(Network n, char* file_name);
Network readNetwork(char* file_name);

#endif
