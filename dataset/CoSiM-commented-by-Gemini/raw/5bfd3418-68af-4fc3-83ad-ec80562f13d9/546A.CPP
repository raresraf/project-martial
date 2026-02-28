/**
 * @file 546A.CPP
 * @brief Iterative solution for the "Soldier and Bananas" problem.
 * @details This program calculates the amount of money a soldier needs to borrow.
 * The cost of bananas increases linearly (1k, 2k, ..., wk). This solution
 * calculates the total cost by iterating from 1 to `w` and summing the cost
 * of each banana. It then determines the deficit.
 *
 * @algorithm Iterative Summation
 * The program uses a for-loop to accumulate the total cost of all `w` bananas.
 * It then compares this total cost with the soldier's initial money `n` to
 * decide if a loan is needed.
 *
 * @complexity
 * - Time: O(W), where W is the number of bananas to buy, due to the single loop.
 * - Space: O(1), for storing a few integer variables.
 */

#include <bits/stdc++.h> // Includes all standard C++ libraries.
using namespace std;

int main() {
  // k: cost of the first banana.
  // n: initial amount of money the soldier has.
  // w: number of bananas he wants to buy.
  // i: loop counter.
  // j: declared but unused.
  // sum: accumulator for the total cost.
  // a: the amount to borrow.
  int k, n, w, i, j, sum = 0, a;
  cin >> k >> n >> w;

  /**
   * @brief Calculate the total cost iteratively.
   * This loop simulates the purchase of each banana from 1 to w.
   * For each banana `i`, its cost (`k * i`) is added to the `sum`.
   */
  // Invariant: `sum` holds the cost of the first `i-1` bananas.
  for (i = 1; i <= w; i++) {
    sum += (k * i);
  }

  /**
   * @brief Determine the amount to borrow.
   * This block checks if the total cost (`sum`) exceeds the money the soldier
   * has (`n`).
   */
  // Pre-condition: Check if the total cost is greater than the available money.
  if (sum > n) {
    // If so, the amount to borrow is the difference.
    a = sum - n;
  } else {
    // If the soldier has enough or more money, he borrows 0.
    // The program prints 0 and exits immediately.
    cout << "0" << endl;
    return 0;
  }

  // If the program reaches this point, it means `sum > n` was true.
  // Print the calculated amount to borrow.
  cout << a << endl;

  return 0; // Indicate successful execution.
}
