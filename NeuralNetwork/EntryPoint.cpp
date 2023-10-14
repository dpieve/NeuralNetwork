#include "NeuralNetwork.hpp"

int main() {
	const Config config = {
		.layer_sizes = { 2, 10, 1 },
		.activation = activation::logistic,
		.cost = cost::mse,
		.random_state = 42,
		.max_iter = 200,
		.tolerance = 0.0001,
		.n_iter_no_change = 10,
		.bias = 0,
		.momentum = 1.0,
		.learning_rate = 1.0,
		.verbose = false
	};

	const std::vector<std::vector<double>> x = { { 0, 0 }, {0, 1}, {1, 0}, {1, 1} };
	const std::vector<std::vector<double>> y = { {0}, {1}, {1}, {0} };

	NeuralNetwork nn{ config };
	nn.fit(x, y);

	const auto predicted = nn.predict(x);

	for (size_t i = 0; const auto& result : predicted) {
		std::cout << std::format("output #{}:", ++i);
		for (const auto p : result)
			std::cout << ' ' << p;
		std::cout << '\n';
	}

	return 0;
}
