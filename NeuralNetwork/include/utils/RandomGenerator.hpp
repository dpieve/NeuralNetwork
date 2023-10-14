#pragma once

#include <random>

template<typename T>
concept floating_point = std::is_floating_point_v<T>;

template <floating_point T>
class RandomGenerator
{
public:
	RandomGenerator(const uint32_t seed, const T min_value, const T max_value)
		: engine_{ seed == 0 ? rd_() : seed }
		, distribution_{ min_value, max_value }
		, min_value_{ min_value }
		, max_value_{ max_value }
	{
	}

	[[nodiscard]] T operator()()
	{
		return distribution_(engine_);
	}

	[[nodiscard]] std::pair<T, T> get_range() const
	{
		return { min_value_, max_value_ };
	}

private:
	std::random_device rd_{};
	std::mt19937_64 engine_;
	std::uniform_real_distribution<T> distribution_;
	T min_value_;
	T max_value_;
};

using Random = RandomGenerator<double>;