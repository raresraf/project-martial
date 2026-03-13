/**
 * @file 546A.CPP
 * @brief Solution to the "Soldier and Bananas" problem from Codeforces.
 *
 * @details
 * This program calculates the amount of money a soldier needs to borrow to buy
 * a certain number of bananas (`w`). The cost increases linearly: the first
 * banana costs `k`, the second `2k`, ..., the i-th costs `i*k`.
 *
 * Problem variable mapping to code:
 * - `k`: Cost of the first banana.
 * - `n`: Soldier's initial money.
 * - `w`: Number of bananas to buy.
 *
 * @algorithm Arithmetic Series Summation (Iterative)
 * 1. Read the initial cost `k`, soldier's money `n`, and number of bananas `w`.
 * 2. Loop from 1 to `w` to simulate buying each banana.
 * 3. In each iteration `i`, add the cost of the current banana (`k * i`) to a running total `sum`.
 * 4. After the loop, if the total `sum` is greater than the soldier's money `n`,
 *    the amount to borrow is `sum - n`.
 * 5. Otherwise, the soldier has enough money, and the amount to borrow is 0.
 *
 * @time_complexity O(W), where W is the number of bananas.
 * @space_complexity O(1).
 *
 * @note The `counter` variable is redundant; the loop variable `i` could be used directly.
 *       An O(1) solution is also possible using the formula for an arithmetic series:
 *       `total_cost = k * w * (w + 1) / 2`.
 */
#include <iostream>
using namespace std;

int main() {
  long long n, k, w; // Use long long to prevent overflow for large w.
  long long sum = 0;
  cin >> k >> n >> w;

  /**
   * @brief This loop calculates the total cost of `w` bananas.
   * The cost of the i-th banana is `i * k`. This is summed up iteratively.
   */
  for (int i = 1; i <= w; i++) {
    sum += k * i;
  }

  /**
   * @brief Determine the amount to borrow.
   * If the total cost `sum` is greater than the initial money `n`,
   * the difference is the amount to borrow. Otherwise, it's 0.
   */
  if (sum > n)
    cout << sum - n << "
";
  else
    cout << "0
";

  return 0;
}
