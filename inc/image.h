#ifndef IMAGE_H_
#define IMAGE_H_

#include <string>
#include <stdio.h>

using namespace std;

class Image{
private:
public:
    virtual ~Image() = 0;
    // virtual void init() = 0;
    // virtual void save() = 0;

};
inline Image::~Image() {}

#endif
