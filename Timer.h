#pragma once
#include <ctime>
#include <chrono>

#ifdef _WIN32
    #define timegm _mkgmtime
#endif

typedef std::chrono::high_resolution_clock clock_;

class Timer
{
public:
	Timer() : start(clock_::now()) {}

	Timer(std::chrono::time_point<clock_>& tp) : start(tp) {}

	void SetStart(std::chrono::time_point<clock_>& tp) { start = tp; }

	void ResetTimer() { start = clock_::now(); }

	int64_t NanoCount() const
	{
		return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_::now() - start).count();
	}

	int64_t MicroCount() const
	{
		return std::chrono::duration_cast<std::chrono::microseconds>(clock_::now() - start).count();
	}

	double MilliCount() const
	{
		return (double)MicroCount() * 1e-3;
	}

	double SecsCount() const
	{
		return (double)MicroCount() * 1e-6;
	}

private:
	std::chrono::time_point<clock_> start;
};
