#pragma once

#include "Matrix.hpp"
#include "configs/ActivationFunctions.hpp"
#include "configs/Config.hpp"

class Layer
{
public:
	Layer(const size_t size, const activation& activation)
		: activation_{ activation }
		, value_{ 1, size }
		, activated_{ 1, size }
		, derived_{ 1, size }
	{
	}

	void set_value(const size_t neuron, const double value)
	{
		value_[0][neuron] = value;

		switch (activation_)
		{
		case activation::logistic:
			const double activated = Logistic::activate(value);
			const double derived = Logistic::derive(activated);
			activated_[0][neuron] = activated;
			derived_[0][neuron] = derived;
			break;
		}
	}

	[[nodiscard]] size_t size() const { return value_.cols(); }
	[[nodiscard]] const Matrix& values() const { return value_; }
	[[nodiscard]] const Matrix& activated() const { return activated_; }
	[[nodiscard]] const Matrix& derived() const { return derived_; }

private:
	activation activation_;
	Matrix value_;
	Matrix activated_;
	Matrix derived_;
};