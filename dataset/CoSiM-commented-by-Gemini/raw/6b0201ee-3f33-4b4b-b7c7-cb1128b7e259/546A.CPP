
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

/**
 * @file 546A.CPP
 * @brief Solution to Codeforces problem 546A, "Soldier and Bananas".
 * @details The program calculates the amount of money a soldier needs to borrow
 * to buy a certain number of bananas. The cost of each banana increases
 * linearly.
 *
 * Algorithm:
 * The total cost of `w` bananas is `k + 2k + ... + wk`, which can be expressed
 * as `k * (1 + 2 + ... + w)`. The sum of the first `w` integers is calculated
 * using the arithmetic series formula: `w * (w + 1) / 2`. This sum is then
 * multiplied by the base cost `k` to get the total cost. Finally, the program
 * calculates the difference between the total cost and the soldier's initial
 * money `n`. If the cost is greater than the money he has, the difference is
 * printed; otherwise, 0 is printed.
 *
 * Time Complexity: O(1), as the solution relies on a direct mathematical formula.
 * Space Complexity: O(1).
 */
int main() {
  // k: cost of the first banana
  // n: initial amount of money the soldier has
  // w: number of bananas to buy
  ll k, n, w;
  cin >> k >> n >> w;

  /**
   * Block Logic: Calculate the total cost using the arithmetic series sum formula.
   *
   * Logic: The sum of costs is 1*k + 2*k + ... + w*k.
   * This is equivalent to k * (1 + 2 + ... + w).
   * The sum of the first w natural numbers is w*(w+1)/2.
   */
  // `ok` stores the sum of the first `w` integers.
  ll ok = (w * (w + 1)) / 2;
  // `res` is the total cost.
  ll res = ok * k;

  /**
   * Block Logic: Determine the amount to borrow.
   *
   * Pre-condition: `res` is the total cost and `n` is the available money.
   * Logic: If the available money `n` is less than the total cost `res`,
   *        the soldier needs to borrow the difference. Otherwise, he needs nothing.
   */
  if (n < res)
    cout << res - n << endl;
  else
    cout << "0" << endl;
}
