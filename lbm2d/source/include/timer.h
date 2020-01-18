/*
 * Timer.cpp
 *
 *  Created on: Nov 5, 2019
 *      Author: yaro
 */

#ifndef _TIMER_H_
#define _TIMER_H_

#include <iostream>
#include <chrono>

struct Timer
{
    Timer(){
		start = std::chrono::steady_clock::now();
	}

    std::chrono::duration<long double> seconds(){
      return std::chrono::duration<long double>(std::chrono::steady_clock::now() - start);
    }
	~Timer(){}
private:
    std::chrono::_V2::steady_clock::time_point start;

};

#endif /*_TIMER_H_ */
