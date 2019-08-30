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
double SignalGenerator::whiteNoiseSample(float snr)
{/* Generates additive white Gaussian Noise samples with zero mean and a standard deviation of 1. */

	double temp1;
	double temp2;
	double result;
	int p;

	float namp = amplitude/sqrt(snr); // noise ampltide
	p = 1;

	while( p > 0 )
	{
		temp2 = ( rand() / ( (double)RAND_MAX ) );
		if ( temp2 == 0 )
		{// temp2 is >= (RAND_MAX / 2)
			p = 1;
		}
		else
		{// temp2 is < (RAND_MAX / 2)
			p = -1;
		}
	}

	temp1 = cos( ( namp * (double)PI_F ) * rand() / ( (double)RAND_MAX ) );
	result = sqrt( -namp * log( temp2 ) ) * temp1;

	return result;	// return the generated random sample to the caller

}// end AWGN_generator()

void SignalGenerator::sweep(float bandwidth, float duration, float fdoppler, bool noise)
{
	allocateMemory();
	float f, n = 0, snr = 2;
	float tstep = duration / length;
	float fstep = bandwidth / duration * tstep;
	float fstart = fc - bandwidth/2;
	float fstop = fc + bandwidth/2;
	int i, j;
	if(noise)
	{
		printf("Insert signal to noise ratio: ");
		scanf("%f", &snr);
	}
	printf("Generating sweep... \n");
	for (j = 0; j < records; j++)
	{
		f = fstart + fdoppler*j;
		for (i = 0; i < length; i++)
		{
			f += fstep;
			if(noise)
				n = whiteNoiseSample(snr);
			p_sig[i + j * length] = (float)amplitude*sin(2*PI_F*f*i/fs) + n;
			// if(i == length -1)
			// 	printf("End: %f; Fstop: %f, fstep: %f; tstep: %f \n", f, fstop, fstep, tstep);
		}
	}
}

void SignalGenerator::noisySinus(float snr)
{
	allocateMemory();
	int i, j;
	double n;
	printf("Generate sinus signal\n");
	for (j = 0; j < records; j++)
		for (i = 0; i < length; i++)
		{
			n = whiteNoiseSample(snr);
			p_sig[i + j * length] = (float)amplitude*sin(2*PI_F*fc*i/fs) + n;
		}
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

void SignalGenerator::save(float bandwidth, float duration)
{
	int ele_size = sizeof(float);
	FILE* fid;
	fid = fopen("./results/data/signal.dat", "wb");
	fwrite((void*)&channels, sizeof(channels), 1, fid);
	fwrite((void*)&records, sizeof(records), 1, fid);
	fwrite((void*)&length, sizeof(length), 1, fid);
	fwrite((void*)&ele_size, sizeof(ele_size), 1, fid);
	fwrite((void*)&fs, sizeof(fs), 1, fid);
	fwrite((void*)&fc, sizeof(fc), 1, fid);
	fwrite((void*)&bandwidth, sizeof(bandwidth), 1, fid);
	fwrite((void*)&duration, sizeof(duration), 1, fid);
	fwrite((void*)getSignal(), sizeof(*p_sig), size/sizeof(*p_sig),fid);
	fclose(fid);
	// printf("Channels = %d\n", channels);
	// printf("Records = %d\n", records);
	// printf("Samples = %d\n", length);
	// printf("Fsample = %f\n", fs);
	// printf("Fcenter = %f\n", fc);
}

void SignalGenerator::load()
{
	FILE* fid;
	fid = fopen("./results/data/signal.dat", "wb");
	fread((void*)&channels, sizeof(channels), 1, fid);
	fread((void*)&records, sizeof(records), 1, fid);
	fread((void*)&length, sizeof(length), 1, fid);
	fread((void*)&fs, sizeof(fs), 1, fid);
	allocateMemory();
	fread((void*)getSignal(), sizeof(*p_sig), size/sizeof(*p_sig),fid);
	fclose(fid);

	printf("Channels = %d\n", channels);
	printf("Records = %d\n", records);
	printf("Samples = %d\n", length);
	printf("Fsample = %f\n", fs);
	printf("Fcenter = %f\n", fc);

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
