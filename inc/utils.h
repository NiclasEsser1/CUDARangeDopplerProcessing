/*
 * utils.h
 *
 *  Created on: Feb 20, 2019
 *      Author: niclas
 */
#ifndef UTIL_H_
#define UTIL_H_
#include <string>
#include <iostream>
#include <algorithm>

using namespace std;

namespace utils
{
    void msg(string str);
    int str2int(string str);
    bool is_number(string str);
}

#endif
