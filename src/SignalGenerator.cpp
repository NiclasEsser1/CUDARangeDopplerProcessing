#include "SignalGenerator.h"
#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <math.h>       /* sin */



SignalGenerator::SignalGenerator(float fs, float fc, int length, int records, int chan)
{
	setCenterFreq(fc);
	setSampleFreq(fs);
	setLength(length);
	setRecords(records);
	setChannels(chan);

	p_sig = NULL;
	allocateMemory();
}


SignalGenerator::~SignalGenerator()
{
	freeBuffer((void**)&p_sig);
}

void SignalGenerator::allocateMemory()
{
	p_sig = (float*)malloc(length*records*channels*sizeof(float));
	CHECK_ALLOCATION(p_sig, __LINE__, __FILE__);
	printf("Allocated memory for signal generation: %ld kBytes\n", (long)length*records*channels*sizeof(float)/1024);
error:
	return;
}

void SignalGenerator::freeBuffer(void** buf)
{
	if(buf)
		free(*buf);
}

void SignalGenerator::sinus()
{
	int i, j;
	printf("Generate sinus signal\n");
	for (j = 1; j <= records; j++)
		for (i = 0; i < length; i++)
			p_sig[i*j] = sin(2*PI_F*fc*i*j/fs);
}

void SignalGenerator::cosinus()
{
	int i, j;
	printf("Generate cosinus signal\n");
	for (j = 1; j <= records; j++)
		for (i = 0; i < length; i++)
			p_sig[i*j] = cos(2 * PI_F*fc*i*j / fs);
}

void SignalGenerator::rectangle()
{
	int i, j, k;
	printf("Generate rectangle signal\n");
	for (j = 1; j <= records; j++)
	{
		for (i = 0; i < length; i++)
		{
			if (i*j % (int)fc == 0)
				toggle(&k);
			if (k)
				p_sig[i*j] = 1;
			else
				p_sig[i*j] = 0;
		}
	}
}

void SignalGenerator::toggle(int *p)
{
	if (*p)
		*p = 0;
	else
		*p = 1;
}

void SignalGenerator::printSignal()
{
	int i, j, k;
	printf("Print samples of signal: \n");
	for(i = 1; i <= channels; i++)
	{
		printf("Channel %d: \n",i);
		for(j = 1; j <= records; j++)
		{
			printf("Record %d: \n",k);
			for(k = 1; k <= length; k++)
			{
				printf("Signal[%d][%d][%d] = %f\n", i,j,k, p_sig[i*j*k]);
			}
		}
	}
}

float* SignalGenerator::getSignal()
{
	return &p_sig[0];
}

void  SignalGenerator::setSignal(float* signal)
{
	p_sig = signal;
}


void SignalGenerator::setCenterFreq(float val)
{
	 fc = val;
}
void SignalGenerator::setSampleFreq(float val)
{
	 fs = val;
}
void SignalGenerator::setChannels(int val)
{
 	channels = val;
}
void SignalGenerator::setRecords(int val)
{
 	records = val;
}
void SignalGenerator::setLength(int val)
{
	length = val;
}
