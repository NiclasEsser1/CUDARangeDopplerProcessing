#ifndef SIGNALGENERATOR_H_
#define SIGNALGENERATOR_H_

#include <string>
#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <math.h>       /* sin */
#include <iostream>
#include "utils.h"

#define CHECK_ALLOCATION(ptr, line, file){if(ptr == NULL){printf("Allocation failed in line %d file: %s \n", line, file);goto error;}}
#define PI_F   3.14159f

using namespace std;
using namespace utils;

class SignalGenerator
{
public:
	SignalGenerator(float fsample=0, float fcenter=0, float amp=1, int x = 1, int y = 1, int z = 1);
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
	float* getSignal(int pos = 0){return &p_sig[pos];}
	size_t getSize(){return size;}
	int getLength(){return length;}
	int getChannels(){return channels;}
	int getTotalRecords(){return total_records;}
	void setCenterFreq(float val);
	void setSampleFreq(float val);
	void setChannels(int val);
	void setRecords(int val);
	void setLength(int val);
	void setSignal(float* signal);
	short* getLoadedSignal(int ch = 0){return loaded_sig[ch];}
	void printSignal();
	void printLoadedSignal();
	void save(std::string filename = "signal.dat", float bandwidth = 0, float duration = 0);
	bool load(std::string filename, int records_to_read);

private:
	float fs;
	float fc;
	float amplitude;
	int channels;
	int records;
	int length;
	int total_records;
	size_t size;
	float* p_sig;
	short** loaded_sig = nullptr;
};

#endif
