/**
 * @file 546A.CPP
 * @brief Solution to the "Soldier and Bananas" problem (546A) from Codeforces.
 * @details This program calculates the amount of money a soldier needs to borrow to buy a
 * specific number of bananas, where the price of each subsequent banana increases linearly.
 * This implementation uses a direct mathematical formula for an O(1) solution.
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm The solution calculates the total cost using the formula for the sum of an
 * arithmetic series, then determines the deficit.
 * 1. Read the initial cost of one banana (k), the soldier's initial money (n), and the
 *    number of bananas to buy (w).
 * 2. The total cost is `k * (1 + 2 + ... + w)`. The sum of the first `w` integers is
 *    calculated with the formula `w * (w + 1) / 2`.
 * 3. The total cost is therefore `k * w * (w + 1) / 2`.
 * 4. The amount to borrow is the total cost minus the initial money `n`. If this
 *    value is negative (the soldier has enough money), the amount to borrow is 0.
 *
 * @complexity
 * Time Complexity: O(1) - The total cost is calculated using a direct mathematical formula,
 * involving only a few arithmetic operations, independent of 'w'.
 * Space Complexity: O(1) - The program uses a fixed number of variables.
 */
#include <iostream>

/**
 * @brief Prints a size_t value to standard output followed by a newline.
 * @param v The value to be printed.
 */
void answer(size_t v) { std::cout << v << '\n'; }

/**
 * @brief Solves the "Soldier and Bananas" problem.
 * @param k The cost of the first banana.
 * @param n The soldier's initial amount of money.
 * @param w The number of bananas to buy.
 */
void solve(unsigned k, unsigned n, unsigned w) {
  // Calculate the sum of the arithmetic series (1 + 2 + ... + w).
  const unsigned sum_i = w * (w + 1) / 2;
  // Calculate the total cost by multiplying the series sum by the base cost k.
  const unsigned cost = sum_i * k;

  // Determine the amount to borrow. If the cost is greater than the available money,
  // the difference is the borrowed amount. Otherwise, it's 0.
  answer(cost > n ? cost - n : 0);
}

int main() {
  unsigned k, n, w;
  std::cin >> k >> n >> w;

  solve(k, n, w);

  return 0;
}
