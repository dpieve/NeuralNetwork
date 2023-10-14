#pragma once

#include <vector>

enum class activation
{
	logistic,
};

enum class cost
{
	mse,
};

struct Config
{
	std::vector<size_t> layer_sizes;
	activation activation;
	cost cost;
	uint32_t random_state;
	uint32_t max_iter;
	double tolerance;
	uint32_t n_iter_no_change;
	double bias;
	double momentum;
	double learning_rate;
	bool verbose;
};