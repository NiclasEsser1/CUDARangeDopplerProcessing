#ifndef SIGNALGENERATOR_H_
#define SIGNALGENERATOR_H_

#define CHECK_ALLOCATION(ptr, line, file){if(ptr == NULL){printf("Allocation failed in line %d file: %s \n", line, file);goto error;}}
#define PI_F   3.14159f


class SignalGenerator
{
public:
	SignalGenerator(float fs, float fc, int length, int records = 1, int chan = 1);
	~SignalGenerator();
	void allocateMemory();
	void freeBuffer(void** buf);
	void sinus();
	void cosinus();
	void rectangle();
	void toggle(int *p);
	float* getSignal();
	void setCenterFreq(float val);
	void setSampleFreq(float val);
	void setChannels(int val);
	void setRecords(int val);
	void setLength(int val);
	void setSignal(float* signal);
	void printSignal();

private:
	float fs;
	float fc;
	int channels;
	int records;
	int length;
	float* p_sig;
};

#endif
