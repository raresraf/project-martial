/**
 * @file 1366A.CPP
 * @brief This program processes pairs of unsigned integers (a, b) and computes a result
 *        based on their relative values and sum.
 *
 * Note: While the filename suggests "Minimal Square" (Codeforces 1366A), the implemented logic in the `solve` function
 *       computes a result following a pattern common in resource distribution or grouping problems, rather than geometric area.
 * Algorithm: For each pair (a, b), the `solve` function first ensures `a <= b`.
 *            If `2 * a <= b`, it returns `a`.
 *            Otherwise (if `a <= b < 2 * a`), it returns `(a + b) / 3`.
 * Time Complexity: O(1) for each test case, as the calculations involve a fixed number of operations.
 * Space Complexity: O(1) for storing a few variables.
 */
#include <iostream> // Required for std::cin, std::cout.
#include <algorithm> // Required for std::swap.

/**
 * @brief Prints the computed unsigned integer result followed by a newline.
 * @param v The unsigned integer value to be printed.
 * Functional Utility: Provides a standardized way to output the answer for each test case.
 */
void answer(unsigned v) { std::cout << v << '\n'; }

/**
 * @brief Computes a result based on two unsigned integers `a` and `b` following specific logic.
 * @param a The first unsigned integer input.
 * @param b The second unsigned integer input.
 * @pre `a` and `b` are non-negative unsigned integers.
 * @post The computed result is printed via the `answer` function.
 * Functional Utility: Encapsulates the core logic for processing each pair of inputs.
 */
void solve(unsigned a, unsigned b) {
  // Block Logic: If 'a' and 'b' are equal, the result is their sum divided by 3.
  if (a == b)
    return answer((a + b) / 3);

  // Block Logic: Ensures that 'a' is always less than or equal to 'b' for consistent logic.
  if (a > b)
    std::swap(a, b);

  // Block Logic: If 'b' is at least twice 'a', the result is 'a'.
  // Invariant: This condition often implies a limiting factor, where 'a' is the maximum possible result.
  if (2 * a <= b)
    return answer(a);

  // Block Logic: Handles the remaining case where 'a <= b < 2 * a'.
  // It computes an intermediate value 'k' and then applies a formula.
  // This simplifies to (a+b)/3.
  const unsigned k = 2 * a - b; // Intermediate calculation.
  answer(2 * k / 3 + a - k);    // The calculation simplifies to (a+b)/3 in this branch.
}

/**
 * @brief Processes a single test case.
 * @pre `std::cin` has two unsigned integers `a` and `b` available.
 * @post The `solve` function is called with `a` and `b`, and its result is printed.
 */
void test_case() {
  unsigned a, b; // Inputs for the current test case.
  // Functional Utility: Reads the two unsigned integers for the current test case.
  std::cin >> a >> b;

  solve(a, b); // Calls the core logic function.
}

/**
 * @brief Main function to handle multiple test cases.
 * @pre Input starts with `t`, the number of test cases, followed by inputs for each `test_case`.
 * @post Executes `test_case()` `t` times.
 */
int main() {
  size_t t; // Variable to store the number of test cases.
  // Functional Utility: Reads the number of test cases.
  std::cin >> t;

  // Block Logic: Loops through each test case.
  while (t-- > 0)
    test_case(); // Calls the function to process a single test case.

  // Indicates successful program execution.
  return 0;
}
