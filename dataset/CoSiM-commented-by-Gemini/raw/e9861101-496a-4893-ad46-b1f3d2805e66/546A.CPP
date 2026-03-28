/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem from Codeforces.
 *
 * A soldier buys `w` bananas. The first costs `k`, the second `2k`, ..., the
 * `w`-th costs `w*k`. Given the soldier has `n` dollars, this program
 * calculates how much money must be borrowed. If the soldier has enough money,
 * the amount to borrow is 0.
 *
 * @algorithm The total cost is the sum of an arithmetic series:
 * k + 2k + ... + w*k = k * (1 + 2 + ... + w).
 * The sum of the first `w` integers is `w*(w+1)/2`.
 * The program calculates the total cost using this formula, subtracts the
 * initial money `n`, and if the result is positive, that is the amount to
 * borrow. Otherwise, the amount is 0.
 *
 * @time_complexity O(1), as the solution is a direct formulaic calculation.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  int k, n, w;
  cin >> k >> n >> w;
  // Calculate the total cost for `w` bananas and subtract the initial money `n`.
  // The cost is k * (1 + 2 + ... + w) = k * w * (w + 1) / 2.
  int ris = k * (w) * (w + 1) / 2 - n;
  
  /**
   * @block_logic If the calculated amount `ris` is positive, it represents the
   * money to be borrowed. Otherwise, the soldier has enough money.
   */
  if (ris > 0)
    cout << ris << "\n";
  else
    cout << 0 << "\n";
  return 0;
}
