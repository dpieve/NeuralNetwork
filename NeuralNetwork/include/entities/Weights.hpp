#pragma once

#include <vector>
#include "Matrix.hpp"
#include "utils/RandomGenerator.hpp"

class Weights
{
public:
	Weights(const std::vector<size_t>& layers_size, const uint32_t seed, const double min_value = 0.0, const double max_value = 1.0)
		: random_{ seed, min_value, max_value }
	{
		for (size_t i = 0; i + 1 < layers_size.size(); ++i)
		{
			const size_t rows = layers_size[i];
			const size_t cols = layers_size[i + 1];

			Matrix weight = generate_random_weight(rows, cols);
			weights_.push_back(weight);
		}
	}

	void set_weights(std::vector<Matrix>&& weights) { weights_ = std::move(weights); }
	[[nodiscard]] const Matrix& back() const { return weights_.back(); }

	Matrix& operator[] (const size_t index)	{ return weights_[index]; }
	const Matrix& operator[] (const size_t index) const	{ return weights_[index]; }

private:
	[[nodiscard]] Matrix generate_random_weight(const size_t rows, const size_t cols)
	{
		Matrix weight{ rows, cols };
		for (size_t r = 0; r < rows; ++r)
		{
			for (size_t c = 0; c < cols; ++c)
			{
				weight[r][c] = random_();
			}
		}
		return weight;
	}

	Random random_;
	std::vector<Matrix> weights_;
};