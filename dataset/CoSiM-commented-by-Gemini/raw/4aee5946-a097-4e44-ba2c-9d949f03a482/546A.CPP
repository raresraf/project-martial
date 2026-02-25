/**
 * @file 546A.CPP
 * @brief Solution to Codeforces problem 546A (Soldier and Bananas).
 * @author Solved by: [User who submitted the code]
 *
 * @details
 * This program calculates the amount of money a soldier needs to borrow to buy
 * a certain number of bananas. The cost of the i-th banana is `i * k`. The
 * soldier wants to buy `w` bananas and has `n` dollars.
 *
 * @section algorithm
 * The total cost to buy `w` bananas is the sum of `k + 2k + 3k + ... + w*k`.
 * This is an arithmetic series which can be expressed as `k * (1 + 2 + ... + w)`.
 * The sum of the first `w` positive integers is given by the formula `w * (w + 1) / 2`.
 *
 * The algorithm is as follows:
 * 1. Read the cost of the first banana (`k`), the soldier's initial money (`n`),
 *    and the number of bananas to buy (`w`).
 * 2. Calculate the total cost using the formula: `total_cost = k * w * (w + 1) / 2`.
 * 3. Determine the amount to borrow. This is the `total_cost` minus the money
 *    the soldier has (`n`).
 * 4. If `total_cost > n`, the amount to borrow is `total_cost - n`.
 * 5. If `total_cost <= n`, the soldier has enough money and needs to borrow 0.
 *
 * This approach is efficient, solving the problem in constant time.
 *
 * @section complexity
 * - Time Complexity: O(1), as the calculation is done using a direct formula
 *   without any loops dependent on the input size.
 * - Space Complexity: O(1).
 */

#include <iostream>
#include <cmath> // For std::abs

using namespace std;

int main() {
  // k: the cost of the first banana.
  // n: the soldier's initial amount of money (long to handle large values).
  // w: the total number of bananas the soldier wants to buy.
  long long k, n, w;
  cin >> k >> n >> w;

  // Calculate the total cost to buy w bananas.
  // This uses the formula for the sum of an arithmetic series.
  long long total_cost = (w * (w + 1) / 2) * k;

  // Calculate the amount the soldier needs to borrow.
  long long borrow_amount = total_cost - n;

  // If the borrow_amount is positive, that's what the soldier needs.
  // Otherwise, the soldier has enough money and needs to borrow 0.
  if (borrow_amount > 0) {
    cout << borrow_amount << endl;
  } else {
    cout << 0 << endl;
  }

  return 0;
}
