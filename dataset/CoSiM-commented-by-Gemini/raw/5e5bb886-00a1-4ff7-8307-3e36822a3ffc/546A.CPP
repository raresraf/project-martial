/**
 * @file 546A.CPP
 * @brief Iterative solution for the "Soldier and Bananas" problem from Codeforces.
 * @details This program calculates the amount of money a soldier needs to borrow to buy `w`
 * bananas, where the cost of the i-th banana is `i*k`. The total cost is calculated
 * using a loop.
 *
 * @algorithm Iterative Summation
 * 1. Read the initial cost `k`, the soldier's money `n`, and the number of bananas `w`.
 * 2. Use a loop to calculate the total cost by summing `i * k` for `i` from 1 to `w`.
 * 3. Subtract the soldier's initial money `n` from the total sum to find the deficit.
 * 4. If the deficit is positive, print it. Otherwise, print 0.
 *
 * @complexity
 * - Time: O(W), where W is the number of bananas, due to the single loop.
 * - Space: O(1), for storing a few variables.
 */

#include <iostream>
// The following includes are not used in this specific solution but are common
// in competitive programming templates.
#include <queue>
#include <stack>
#include <string>
#include <vector>

using namespace std;

int main() {
  // k: the cost of the first banana.
  // n: the initial amount of money the soldier has.
  // w: the number of bananas to buy.
  // sum: used to accumulate the total cost.
  long long k, n, w, sum = 0;
  
  cin >> k >> n >> w;

  /**
   * @brief Calculate the total cost of all `w` bananas iteratively.
   * The cost of the i-th banana is `i * k`.
   */
  // Invariant: `sum` holds the total cost of the first `i-1` bananas.
  for (int i = 1; i <= w; i++) {
    sum = sum + i * k;
  }
  
  // Repurpose the `sum` variable to now hold the deficit (amount to borrow).
  sum = sum - n;

  /**
   * @brief Print the required amount to borrow.
   * If the calculated deficit (`sum`) is zero or negative, the soldier has enough
   * money, so they need to borrow 0. Otherwise, print the deficit.
   */
  if (sum <= 0) {
    cout << 0 << endl;
  } else {
    cout << sum << endl;
  }
  
  return 0; // Indicate successful execution.
}
