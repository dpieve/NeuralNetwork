#pragma once

#include "configs/Config.hpp"
#include "entities/Error.hpp"
#include "entities/Layers.hpp"
#include "entities/Matrix.hpp"
#include "entities/Weights.hpp"

class NeuralNetwork
{
public:
	explicit NeuralNetwork(const Config& config);
	void fit(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets);
	[[nodiscard]] std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& inputs);

private:
	void feed_forward();
	void updated_errors();
	void backpropagation();

	[[nodiscard]] Matrix output_gradients();
	[[nodiscard]] Matrix last_hidden_to_output_weights(const Matrix& gradients) const;
	[[nodiscard]] Matrix calculate_updated_weights(const Matrix& weights, const Matrix& deltas) const;

	void print_layers_and_weights() const;
	void print_target() const;

	Config config_;
	Layers layers_;
	Weights weights_;
	Error error_;
	std::vector<double> targets_;
};