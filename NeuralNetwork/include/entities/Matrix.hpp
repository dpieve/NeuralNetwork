#pragma once

#include <vector>
#include <iomanip>
#include <iostream>

class Matrix
{
public:
	using Row = std::vector<double>;
	using Rows = std::vector<Row>;

	Matrix(const size_t rows, const size_t cols)
	{
		matrix_.assign(rows, std::vector(cols, 0.0));
	}

	explicit Matrix(Rows matrix)
		: matrix_{ std::move(matrix) }
	{
	}

	[[nodiscard]] Matrix transpose() const
	{
		const auto rows = matrix_.size();
		const auto cols = matrix_[0].size();

		auto transposed = Matrix(std::vector(cols, Row(rows, 0)));

		for (size_t i = 0; i < matrix_.size(); ++i)
		{
			for (size_t j = 0; j < matrix_[0].size(); ++j)
			{
				transposed[j][i] = matrix_[i][j];
			}
		}

		return transposed;
	}

	Matrix operator * (const Matrix& b) const
	{
		auto [rows_a, cols_a] = size();
		auto [rows_b, cols_b] = b.size();

		Matrix prod(rows_a, cols_b);

		for (size_t i = 0; i < rows_a; ++i) {
			for (size_t j = 0; j < cols_b; ++j) {
				double sum = 0.0;
				for (size_t k = 0; k < rows_b; ++k) {
					sum += (matrix_[i][k] * b[k][j]);
				}
				prod[i][j] = sum;
			}
		}

		return prod;
	}

	[[nodiscard]] std::pair<size_t, size_t> size() const {	return { rows(), cols() }; }
	[[nodiscard]] size_t rows() const noexcept { return matrix_.size(); }
	[[nodiscard]] size_t cols() const { return matrix_[0].size(); }

	Row& operator[] (const size_t row) { return matrix_[row]; }
	const Row& operator[] (const size_t row) const { return matrix_[row]; }

	void print() const
	{
		for (const auto& row : matrix_) {
			for (const auto& v : row)
				std::cout << std::fixed << std::setprecision(2) << v << '\t';
			std::cout << '\n';
		}
	}

private:
	Rows matrix_;
};