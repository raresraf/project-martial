/**
 * @file 1140D.CPP
 * @brief Solution to a problem involving the computation of a specific mathematical sum.
 *
 * This program takes an unsigned integer 'n' as input. It then calculates
 * the sum of the expression `x * (x + 1)` for all integer values of 'x'
 * starting from 2 up to, but not including, 'n'. The final computed sum is
 * then printed to the console. This problem focuses on basic iterative summation.
 *
 * Algorithm: Iterative summation of terms `x * (x + 1)`.
 * Time Complexity: O(N), where N is the input integer, as the loop runs approximately N times.
 * Space Complexity: O(1) for variable storage.
 */
#include <iostream>

/**
 * @brief Prints an unsigned integer followed by a newline character.
 * @param v The unsigned integer value to be printed.
 * Functional Utility: Provides a standardized way to output the final result.
 */
void answer(unsigned v) { std::cout << v << '\n'; }

/**
 * @brief Computes the sum of `x * (x + 1)` for x from 2 to n-1.
 * @param n The upper bound (exclusive) for the summation.
 * Functional Utility: Encapsulates the core logic for the summation task.
 */
void solve(unsigned n) {
  unsigned s = 0; // Initialize sum 's' to 0.
  /**
   * Block Logic: Iterates through values of 'x' from 2 up to 'n-1'.
   * Invariant: In each iteration, `s` accumulates the term `x * (x + 1)`.
   * Pre-condition: 'n' is an unsigned integer.
   */
  for (unsigned x = 2; x < n; ++x)
    // Add the current term `x * (x + 1)` to the running sum 's'.
    s += x * (x + 1);

  // Call the answer function to print the final sum.
  answer(s);
}

int main() {
  unsigned n; // Declare an unsigned integer 'n' for input.
  // Read the input value for 'n'.
  std::cin >> n;

  // Call the solve function to perform the calculation and print the result.
  solve(n);

  return 0; // Indicate successful program execution.
}
