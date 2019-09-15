#ifndef SIGNALGENERATOR_H_
#define SIGNALGENERATOR_H_

#include <string>
#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <math.h>       /* sin */

#define CHECK_ALLOCATION(ptr, line, file){if(ptr == NULL){printf("Allocation failed in line %d file: %s \n", line, file);goto error;}}
#define PI_F   3.14159f



class SignalGenerator
{
public:
	SignalGenerator(float fsample, float fcenter, float amp, int x, int y = 1, int z = 1);
	~SignalGenerator();
	void allocateMemory();
	void freeBuffer(float* buf);
	double whiteNoiseSample(float snr = 2);
	void sweep(float bandwidth, float duration, float fdoppler = 0, bool noise = false, int runs = 1);
	void noisySinus(float snr = 1);
	void sinus();
	void cosinus();
	void rectangle();
	void toggle(int *p);
	float* getSignal(int pos = 0);
	size_t getSize(){return size;}
	void setCenterFreq(float val);
	void setSampleFreq(float val);
	void setChannels(int val);
	void setRecords(int val);
	void setLength(int val);
	void setSignal(float* signal);
	void printSignal();
	void save(std::string filename = "signal.dat", float bandwidth = 0, float duration = 0);
	void load();

private:
	float fs;
	float fc;
	float amplitude;
	int channels;
	int records;
	int length;
	size_t size;
	float* p_sig;
};

#endif
