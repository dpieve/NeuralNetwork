#pragma once

struct Logistic
{
	static double activate(const double value)
	{
		return value / (1 + abs(value));
	}

	static double derive(const double activated)
	{
		return activated * (1 - activated);
	}
};