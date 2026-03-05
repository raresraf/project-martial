/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem.
 *
 * This program calculates the amount of money a soldier needs to borrow to buy a
 * certain number of bananas. The cost of the bananas increases linearly: the
 * first banana costs `k`, the second `2k`, and so on, up to the `w`-th banana
 * which costs `w*k`. The soldier initially has `n` dollars.
 *
 * @algorithm
 * 1. Read the input values:
 *    `k`: the cost of the first banana.
 *    `n`: the initial amount of money the soldier has.
 *    `w`: the total number of bananas the soldier wants to buy.
 *
 * 2. Calculate the total cost of `w` bananas. The total cost is the sum of an
 *    arithmetic series: `k + 2k + 3k + ... + w*k`.
 *    This can be factored as `k * (1 + 2 + 3 + ... + w)`.
 *
 * 3. The sum of the first `w` positive integers is given by the formula: `w * (w + 1) / 2`.
 *
 * 4. Therefore, the total cost is `k * (w * (w + 1) / 2)`. This is calculated and
 *    stored in a `long long` variable `total` to prevent potential integer overflow.
 *
 * 5. Compare the `total` cost with the soldier's money `n`.
 *    a. If `total <= n`, the soldier has enough money. The amount to borrow is 0.
 *    b. If `total > n`, the soldier needs to borrow the difference: `total - n`.
 *
 * 6. Print the amount to be borrowed.
 *
 * @complexity
 * Time Complexity: O(1), as the calculation is done using a direct mathematical formula.
 * Space Complexity: O(1).
 */

#include <bits/stdc++.h>
using namespace std;
// Use a type alias for long long for convenience and to handle potentially large numbers.
typedef long long ll;

int main() {
  // k: cost of the first banana
  // n: initial money
  // w: number of bananas to buy
  int k, n, w;
  cin >> k >> n >> w;

  // Calculate the total cost of `w` bananas.
  // The sum of the series 1 + 2 + ... + w is w*(w+1)/2.
  // The total cost is k * (sum of the series).
  ll total = (ll)w * (w + 1) / 2 * k;

  // Check if the soldier has enough money.
  if (total <= n) {
    // If the cost is less than or equal to the money he has, he borrows nothing.
    cout << 0;
  } else {
    // Otherwise, the amount to borrow is the difference.
    cout << total - n;
  }
  return 0;
}
