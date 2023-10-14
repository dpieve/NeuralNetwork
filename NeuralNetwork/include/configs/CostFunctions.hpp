#pragma once

#include <utility>

struct MseCost
{
	static std::pair<double, double> calculate(const double output, const double target)
	{
		const double dif = abs(target - output);
		const double error = 0.5 * dif * dif;
		const double derived = output - target;
		return std::make_pair(error, derived);
	}
};