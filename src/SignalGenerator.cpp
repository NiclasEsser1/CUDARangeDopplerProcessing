#include "SignalGenerator.h"
#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <math.h>       /* sin */



SignalGenerator::SignalGenerator(float fsample, float fcenter, float amp, int x, int y, int z)
{
	fc = fcenter;
	fs = fsample;
	length = x;
	records = y;
	channels = z;
	amplitude = amp;
	size = x*y*z*sizeof(*p_sig);
	p_sig = NULL;
}


SignalGenerator::~SignalGenerator()
{
	// freeBuffer(p_sig);
}

void SignalGenerator::allocateMemory()
{
printf("SIZE: %ld \n", size);
	p_sig = (float*)malloc(size);
	CHECK_ALLOCATION(p_sig, __LINE__, __FILE__);
	printf("Allocated memory for signal: %lf MBytes\n", (float)size/(1024*1024));
error:
	return;
}

void SignalGenerator::freeBuffer(float* buf)
{
	if(buf != NULL)
		free(buf);
}

void SignalGenerator::sinus()
{
	allocateMemory();
	int i, j;
	printf("Generate sinus signal\n");
	for (j = 0; j < records; j++)
		for (i = 0; i < length; i++)
			p_sig[i + j * length] = (float)amplitude*sin(2*PI_F*fc*i/fs);
}

void SignalGenerator::cosinus()
{
	allocateMemory();
	int i, j;
	printf("Generate cosinus signal\n");
	for (j = 0; j < records; j++)
		for (i = 0; i < length; i++)
			p_sig[i + j * length] = amplitude*cos(2 * PI_F*fc*i / fs);
}

void SignalGenerator::rectangle()
{
	allocateMemory();
	int i, j, k = 0;
	printf("Generate rectangle signal\n");
	for (j = 0; j < records; j++)
	{
		for (i = 0; i < length; i++)
		{
			if ((int)(i*fs) % (int)fc/2 == 0)
				toggle(&k);
			if (k)
				p_sig[i + j * length] = amplitude;
			else
				p_sig[i + j * length] = amplitude-1;
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
	for(i = 0; i < channels; i++)
	{
		printf("Channel %d: \n",i);
		for(j = 0; j < records; j++)
		{
			printf("Record %d: \n",k);
			for(k = 0; k < length; k++)
			{
				printf("Signal[%d][%d][%d] = %f\n", i,j,k, p_sig[i*records*length + j * length + k]);
			}
		}
	}
}

void SignalGenerator::save()
{
	FILE* fid;
	fid = fopen("./results/data/signal.dat", "wb");
	fwrite((void*)&channels, sizeof(channels), 1, fid);
	fwrite((void*)&records, sizeof(records), 1, fid);
	fwrite((void*)&length, sizeof(length), 1, fid);
	fwrite((void*)&fs, sizeof(fs), 1, fid);
	fwrite((void*)getSignal(), sizeof(*p_sig), size/sizeof(*p_sig),fid);
	printf("Channels = %d\n", channels);
	printf("Records = %d\n", records);
	printf("Samples = %d\n", length);
	printf("Fsample = %f\n", fs);

}

float* SignalGenerator::getSignal(int pos)
{
	return &p_sig[pos];
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
