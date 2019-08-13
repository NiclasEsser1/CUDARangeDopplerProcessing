#ifndef SIGNALGENERATOR_H_
#define SIGNALGENERATOR_H_

#define CHECK_ALLOCATION(ptr, line, file){if(ptr == NULL){printf("Allocation failed in line %d file: %s \n", line, file);goto error;}}
#define PI_F   3.14159f


class SignalGenerator
{
public:
	SignalGenerator(float fsample, float fcenter, float amp, int x, int y = 1, int z = 1);
	~SignalGenerator();
	void allocateMemory();
	void freeBuffer(float* buf);
	void sinus();
	void cosinus();
	void rectangle();
	void toggle(int *p);
	float* getSignal(int pos = 0);
	void setCenterFreq(float val);
	void setSampleFreq(float val);
	void setChannels(int val);
	void setRecords(int val);
	void setLength(int val);
	void setSignal(float* signal);
	void printSignal();
	void save();

private:
	float fs;
	float fc;
	float amplitude;
	int channels;
	int records;
	int length;
	float* p_sig;
};

#endif
