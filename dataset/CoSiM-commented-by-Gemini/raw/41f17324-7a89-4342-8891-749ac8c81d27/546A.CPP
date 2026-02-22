/**
 * @file 546A.CPP
 * @brief Solution to Codeforces problem 546A - Soldier and Bananas.
 *
 * @algorithm
 * The problem requires calculating the amount of money a soldier needs to borrow to
 * buy a certain number of bananas. The cost of the bananas increases linearly: the
 * first costs `k`, the second `2k`, ..., up to the `w`-th banana costing `w*k`.
 * The soldier has `n` dollars.
 *
 * The total cost is the sum of an arithmetic series: `k + 2k + ... + wk`.
 * This can be factored to `k * (1 + 2 + ... + w)`.
 * The sum of the first `w` integers is `w * (w + 1) / 2`.
 * Therefore, the total cost is `k * w * (w + 1) / 2`.
 *
 * The algorithm is as follows:
 * 1. Read the cost of the first banana `k`, the initial money `n`, and the
 *    number of bananas to buy `w`.
 * 2. Calculate the total cost using the formula `(k * w * (w + 1)) / 2`.
 * 3. The amount to borrow is the total cost minus the initial money.
 * 4. If the result is positive, that's the amount to borrow. If it's zero or
 *    negative (meaning the soldier has enough money), the amount to borrow is 0.
 *
 * @complexity
 * - Time Complexity: O(1), as the solution relies on a direct mathematical formula.
 * - Space Complexity: O(1), as it only uses a few variables.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {

/**
 * @brief This block is a common competitive programming setup.
 * It redirects standard input and output to files when the code is run locally
 * (i.e., when ONLINE_JUDGE is not defined). It has no effect when submitted
 * to an online judge.
 */
#ifndef ONLINE_JUDGE
  // for getting input from input.txt
  freopen("input.txt", "r", stdin);
  // for writing output to output.txt
  freopen("output.txt", "w", stdout);
#endif

  long long int k, n, w, sum;
  cin >> k >> n >> w;
  // Calculate the total cost of `w` bananas.
  // This is k * (1 + 2 + ... + w), which simplifies to k * w * (w+1) / 2.
  sum = (k * (w + 1) * (w)) / 2;

  /**
   * @brief Calculate the amount to borrow.
   * If the cost (`sum`) is greater than the money the soldier has (`n`),
   * the difference is the amount to borrow. Otherwise, he needs to borrow 0.
   */
  if (sum - n >= 0) {
    cout << sum - n;
  } else {
    cout << "0";
  }

  return 0;
}