/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem to find the amount of money to borrow.
 *
 * This program calculates the debt a soldier incurs after buying a certain
 * number of bananas, where each subsequent banana has a higher cost.
 *
 * Variable Mapping Note:
 * The variable names in this code do not match the problem statement.
 * - `n` in code is `k` in the problem (cost of the first banana).
 * - `k` in code is `n` in the problem (the soldier's initial money).
 * - `w` in code is `w` in the problem (the number of bananas to buy).
 *
 * Algorithm:
 * 1. The total cost is the sum of an arithmetic progression: `n*1 + n*2 + ... + n*w`.
 * 2. This is calculated using the formula for the sum of the first `w` integers:
 *    Total Cost (`a`) = `n * (w * (w + 1) / 2)`.
 * 3. The amount to borrow (`b`) is the `Total Cost - k` (initial money).
 * 4. If the result is negative (the soldier has enough money), 0 is printed.
 *    Otherwise, the calculated positive amount `b` is printed.
 *
 * Time Complexity: O(1), as it's a direct calculation.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;

#define end return 0 // Unconventional macro for `return 0`

typedef long long ll;
typedef long long int lli;
typedef unsigned long long int ulli;

int main(int argc, char const *argv[]) {
  // Fast I/O setup
  ios_base::sync_with_stdio(0);
  lli n, k, w, a, b, c;

  // Read problem variables with confusing names.
  cin >> n >> k >> w; // n=cost_first, k=initial_money, w=num_bananas

  // `a` is the total cost of all `w` bananas.
  a = (w * (w + 1) / 2) * n;

  // `b` is the amount to borrow (total cost - initial money).
  b = a - k;

  /**
   * @brief Check if the soldier needs to borrow money.
   * If the total cost `a` is less than or equal to the initial money `k`,
   * the difference `b` will be <= 0, so no money is needed.
   */
  if (a - k < 0) {
    cout << 0 << endl;
  } else {
    cout << b << endl;
  }
  return 0;
}
