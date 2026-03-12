/**
 * @file 546A.CPP
 * @brief Solution to the "Soldier and Bananas" problem from Codeforces.
 *
 * This program calculates the amount of money a soldier needs to borrow to buy
 * a certain number of bananas. The cost of the bananas increases linearly: the
 * first costs `k`, the second `2k`, and so on.
 *
 * Problem variable mapping to code:
 * - Cost of the first banana: `k`
 * - Soldier's initial money: `d`
 * - Number of bananas to buy: `n`
 *
 * @algorithm Arithmetic Series Summation
 * @time_complexity O(N), where N is the number of bananas to buy. The cost is calculated with a simple loop.
 * @space_complexity O(1).
 * @note The total cost can also be calculated in O(1) using the formula for an arithmetic series: cost = k * n * (n + 1) / 2.
 */
#include <iostream>

using namespace std;

int main() {
  long long k, d, n, b = 0; // k: base cost, d: initial money, n: banana count, b: total cost
  cin >> k >> d >> n;

  /**
   * @brief Calculates the total cost for `n` bananas.
   * The cost of the i-th banana is `i * k`. This loop sums the cost
   * for all bananas from 1 to `n`.
   */
  for (int i = 1; i <= n; i++)
    b += i * k;

  /**
   * @brief Calculates and prints the amount of money to borrow.
   * If the total cost `b` is greater than the soldier's money `d`, it prints
   * the difference. Otherwise, it prints 0.
   */
  cout << (b - d > 0 ? b - d : 0);
  return 0;
}
