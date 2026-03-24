/**
 * @file 546A.CPP
 * @brief Solution for Codeforces Problem 546A - "Soldier and Bananas".
 * @details This program calculates the amount of money a soldier needs to borrow to buy 'w' bananas,
 * where the cost of each banana increases linearly. The first banana costs 'k', the second '2k', and so on.
 *
 * @author (unknown)
 *
 * @algorithm
 * The core logic is as follows:
 * 1. Read the cost of the first banana (`k`), the soldier's initial money (`n`), and the number of bananas to buy (`w`).
 * 2. Iteratively calculate the total `cost` of `w` bananas by summing the increasing cost of each banana (`i * k`).
 * 3. Compare the total `cost` with the available money `n`.
 * 4. If the soldier has more money than the total cost, the amount to borrow is 0.
 * 5. Otherwise, the amount to borrow is the difference, `cost - n`.
 * 6. The final amount is printed.
 *
 * @complexity
 * Time Complexity: O(w) because of the loop to sum the costs. A more optimal O(1) solution exists
 * by using the arithmetic series formula: `total_cost = k * w * (w + 1) / 2`.
 * Space Complexity: O(1).
 */
#include <iostream>

using namespace std;

int main() {
  /**
   * Functional Utility: Decouples C++ streams from C streams and unties cin from cout
   * for faster I/O, a common optimization in competitive programming.
   */
  ios::sync_with_stdio(0);
  cin.tie(0);

  int k, n, w;
  // k: cost of the first banana.
  // n: soldier's initial money.
  // w: number of bananas to buy.
  cin >> k >> n >> w;

  int cost = 0;
  /**
   * Block Logic: This loop calculates the total cost of buying 'w' bananas.
   * The cost of the i-th banana is `i * k`. The sum is accumulated in the `cost` variable.
   * The total cost can be large, but for the given constraints of the problem, an `int` is sufficient.
   */
  for (int i = 1; i <= w; i++) {
    cost += i * k;
  }

  /**
   * Block Logic: Determines and prints the amount of money the soldier needs to borrow.
   * If the `cost` is less than the money the soldier `n` has, they don't need to borrow anything.
   * Otherwise, the amount to borrow is the difference `cost - n`.
   */
  if (cost < n) {
    cout << 0 << "\n";
  } else {
    cout << cost - n << "\n";
  }

  return 0;
}
