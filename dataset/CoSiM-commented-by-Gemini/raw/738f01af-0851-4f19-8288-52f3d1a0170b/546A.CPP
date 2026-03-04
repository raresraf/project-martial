/**
 * @file 546A.CPP
 * @brief Solution for Codeforces Problem 546A - "Soldier and Bananas".
 *
 * This program attempts to solve the "Soldier and Bananas" problem. The goal
 * is to calculate how much money a soldier needs to borrow. However, this
 * implementation contains a logical flaw where the roles of the initial money (n)
 * and the number of bananas (w) are swapped in the calculation.
 *
 * @algorithm Direct calculation (with a logical error).
 * @time_complexity O(1).
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  // k: cost of the first banana
  // w: soldier's initial money (intended to be 'n')
  // n: number of bananas to buy (intended to be 'w')
  long long k, w, n;
  cin >> k >> w >> n;

  /**
   * @brief This calculation is logically incorrect for the problem statement.
   * It uses 'n' (read as the number of bananas) to calculate the total cost,
   * instead of 'w' (read as the initial money). The correct formula for the
   * total cost is `k * w * (w + 1) / 2`. This code calculates `k * n * (n + 1) / 2`.
   */
  long long ans = n * (n + 1) / 2 * k;

  /**
   * @brief This comparison is also incorrect.
   * It compares the calculated cost 'ans' with 'w' (the initial money),
   * instead of 'n'.
   */
  if (ans <= w)
    cout << 0 << endl;
  else
    cout << ans - w << endl;
  return 0;
}