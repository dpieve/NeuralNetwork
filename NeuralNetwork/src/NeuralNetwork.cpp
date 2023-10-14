#include "NeuralNetwork.hpp"
#include "utils/Stopwatch.hpp"

NeuralNetwork::NeuralNetwork(const Config& config)
	: config_{ config }
	, layers_{ config.layer_sizes, config.activation }
	, weights_{ config.layer_sizes, config.random_state }
	, error_{ config_.cost }
{
}

void NeuralNetwork::fit(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets)
{
	Stopwatch stopwatch{};

	uint32_t iter_no_change = 0;

	for (uint32_t epoch = 0; epoch < config_.max_iter; ++epoch)
	{
		for (size_t i = 0; i < inputs.size(); ++i)
		{
			layers_.set_input(inputs[i]);
			targets_ = targets[i];

			feed_forward();
			updated_errors();
			backpropagation();

			if (config_.verbose)
				print_layers_and_weights();
		}

		if (error_.total() <= config_.tolerance)
		{
			++iter_no_change;
		}
		else
		{
			iter_no_change = 0;
		}

		if (iter_no_change >= config_.n_iter_no_change)
		{
			break;
		}
	}

	if (config_.verbose)
		print_target();
}

std::vector<std::vector<double>> NeuralNetwork::predict(const std::vector<std::vector<double>>& inputs)
{
	std::vector<std::vector<double>> outputs;

	for (const auto& input : inputs)
	{
		layers_.set_input(input);
		feed_forward();

		const auto& output = (layers_.output_layer().activated())[0];
		outputs.push_back(output);
	}

	return outputs;
}

void NeuralNetwork::feed_forward()
{
	for (size_t i = 0; i < layers_.size() - 1; ++i)
	{
		const Matrix& layer = (i == 0 ? layers_[i].values() : layers_[i].activated());
		const Matrix& weight = weights_[i];

		auto next_layer = (layer * weight)[0];

		for (size_t neuron = 0; neuron < next_layer.size(); ++neuron)
		{
			layers_[i + 1].set_value(neuron, next_layer[neuron] + config_.bias);
		}
	}
}

void NeuralNetwork::updated_errors()
{
	const auto& output_layer = (layers_.output_layer().activated())[0];

	double total_error = error_.calculate(output_layer, targets_);

	if (config_.verbose)
		std::cout << std::format("Error={}\n", total_error);
}

Matrix NeuralNetwork::output_gradients()
{
	const Layer& output_layer = layers_.output_layer();
	auto derived_output_layer = output_layer.derived();
	const auto derived_errors = error_.derived();

	Matrix gradients(1, output_layer.size());
	for (size_t i = 0; i < output_layer.size(); ++i)
	{
		const double e = derived_errors[i];
		const double d = derived_output_layer[0][i];
		const double g = e * d;
		gradients[0][i] = g;
	}

	return gradients;
}

Matrix NeuralNetwork::calculate_updated_weights(const Matrix& weights, const Matrix& deltas) const
{
	auto [rows, cols] = weights.size();

	Matrix new_weights(rows, cols);

	for (size_t r = 0; r < rows; ++r)
	{
		for (size_t c = 0; c < cols; ++c)
		{
			const double weight = config_.momentum * weights[r][c];
			const double delta = config_.learning_rate * deltas[r][c];
			new_weights[r][c] = weight - delta;
		}
	}

	return new_weights;
}

Matrix NeuralNetwork::last_hidden_to_output_weights(const Matrix& gradients) const
{
	const Matrix hidden_layer_activated = layers_.last_hidden_layer().activated();
	const Matrix deltas = (gradients.transpose() * hidden_layer_activated).transpose();

	const Matrix& last_weight = weights_.back();

	return calculate_updated_weights(last_weight, deltas);
}

void NeuralNetwork::backpropagation()
{
	Matrix gradients = output_gradients();
	std::vector new_weights{ last_hidden_to_output_weights(gradients) };

	for (auto i = layers_.size() - 2; i > 0; --i)
	{
		gradients = gradients * (weights_[i].transpose());

		Matrix derived_layer = layers_[i].derived();

		for (size_t j = 0; j < derived_layer.cols(); ++j)
		{
			gradients[0][j] = gradients[0][j] * derived_layer[0][j];
		}

		const Matrix& activated = (i == 1) ? layers_[0].values() : layers_[i - 1].activated();
		Matrix activated_transposed = activated.transpose();
		Matrix delta_weights = activated_transposed * gradients;

		Matrix new_weight = calculate_updated_weights(weights_[i - 1], delta_weights);
		new_weights.push_back(new_weight);
	}

	std::ranges::reverse(new_weights.begin(), new_weights.end());

	weights_.set_weights(std::move(new_weights));
}

void NeuralNetwork::print_layers_and_weights() const
{
	for (size_t i = 0; i < layers_.size(); ++i)
	{
		std::string kind;
		if (i == 0)
			kind = "Input";
		else if (i + 1 == layers_.size())
			kind = "Output";
		else
			kind = "Hidden";

		std::cout << std::format("{} Layer {}:\n", kind, i + 1);

		auto layer = (i == 0 ? layers_[i].values() : layers_[i].activated());
		layer.print();

		if (i + 1 < layers_.size())
		{
			std::cout << std::format("Weights {}:\n", i + 1);
			weights_[i].print();
		}
		std::cout << '\n';
	}
}

void NeuralNetwork::print_target() const
{
	std::cout << "Targets:\n";
	for (const double x : targets_)
		std::cout << x << ' ';
	std::cout << '\n';
}