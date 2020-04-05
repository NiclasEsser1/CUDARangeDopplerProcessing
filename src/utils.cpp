#include "utils.h"

namespace utils
{
    void msg(string str)
    {
        cout << str << endl;
    }
    int str2int(string str)
    {
        try
    	{
    		int i = std::stoi(str);
    	}
    	catch (std::invalid_argument const &e)
    	{
    		msg("Bad input: std::invalid_argument thrown");
    	}
    	catch (std::out_of_range const &e)
    	{
    		msg("Integer overflow: std::out_of_range thrown");
    	}
    }
    bool is_number(const string& s)
    {
        return !s.empty() && find_if(s.begin(),
            s.end(), [](unsigned char c) { return !isdigit(c); }) == s.end();
    }
}
