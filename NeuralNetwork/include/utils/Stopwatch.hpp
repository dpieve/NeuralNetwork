#pragma once

#include <chrono>
#include <iostream>

class Stopwatch
{
public:
    Stopwatch() = default;

    Stopwatch(const Stopwatch&) = delete;
    Stopwatch(Stopwatch&&) = delete;
    Stopwatch& operator=(const Stopwatch&) = delete;
    Stopwatch& operator=(Stopwatch&&) = delete;

    ~Stopwatch()
    {
        const TimePoint end_time = Clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_);
        std::cout << "Elapsed time: " << duration.count() << "ms\n";
    }
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    TimePoint start_time_ = Clock::now();
};