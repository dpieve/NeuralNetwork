#pragma once

#include <functional>
#include <vector>
#include <utility>
#include "configs/Config.hpp"
#include "configs/CostFunctions.hpp"

class Error
{
public:
	explicit Error(const cost& cost)
		: cost_{ cost }
	{
	}

	double calculate(const std::vector<double>& outputs, const std::vector<double>& targets)
	{
		std::function<std::pair<double, double>(double, double)> calculate;

		switch (cost_)
		{
		case cost::mse:
			calculate = MseCost::calculate;
			break;
		}

		double total = 0.0;

		for (size_t i = 0; i < outputs.size(); ++i)
		{
			auto [error, derived] = calculate(outputs[i], targets[i]);

			if (i >= errors_.size())
				errors_.push_back(error);
			else
				errors_[i] = error;

			if (i >= derived_.size())
				derived_.push_back(derived);
			else
				derived_[i] = derived;

			total += error;
		}
		totals_.push_back(total);

		errors_.resize(outputs.size());
		derived_.resize(outputs.size());

		return total;
	}

	[[nodiscard]] double total() const noexcept { return totals_.empty() ? 1.0 : totals_.back(); }
	[[nodiscard]] const std::vector<double>& errors() const { return errors_; }
	[[nodiscard]] const std::vector<double>& derived() const { return derived_; }
	[[nodiscard]] const std::vector<double>& totals() const { return totals_; }

private:
	cost cost_;
	std::vector<double> errors_;
	std::vector<double> derived_;
	std::vector<double> totals_;
};