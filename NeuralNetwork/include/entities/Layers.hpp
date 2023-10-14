#pragma once

#include "Layer.hpp"

class Layers
{
public:
	explicit Layers(const std::vector<size_t>& layers_sizes, activation activation)
	{
		for (auto size : layers_sizes)
			layers_.emplace_back(size, activation);
	}

	void set_input(const std::vector<double>& input)
	{
		for (size_t i = 0; i < layers_[0].size(); ++i)
		{
			layers_[0].set_value(i, input[i]);
		}
	}

	[[nodiscard]] const Layer& last_hidden_layer() const { return layers_[size() - 2]; }
	[[nodiscard]] Layer& output_layer() { return layers_[size() - 1]; }

	[[nodiscard]] size_t size() const noexcept { return layers_.size(); }

	Layer& operator[] (const size_t index) { return layers_[index]; }
	const Layer& operator[] (const size_t index) const { return layers_[index]; }

private:
	std::vector<Layer> layers_;
};