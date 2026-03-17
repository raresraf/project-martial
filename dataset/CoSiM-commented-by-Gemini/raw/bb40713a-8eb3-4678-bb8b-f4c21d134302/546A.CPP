/**
 * @file 546A.CPP
 * @brief Solution to the "Soldier and Bananas" problem.
 *
 * @details This program calculates the amount of money a soldier needs to borrow to buy a
 * certain number of bananas. The cost of the bananas increases arithmetically.
 *
 * @section
 * Problem variable mapping:
 * - `k`: Cost of the first banana.
 * - `s`: Initial money the soldier has (in the problem statement, this is usually 'n').
 * - `n`: Number of bananas the soldier wants to buy (in the problem statement, this is usually 'w').
 *
 * Algorithm:
 * 1. Calculate the total cost of 'n' bananas. The cost is an arithmetic series:
 *    k*1 + k*2 + ... + k*n = k * (1 + 2 + ... + n).
 *    This is calculated using the formula for the sum of the first n integers: k * n * (n+1) / 2.
 * 2. Compare the total cost with the money the soldier has ('s').
 * 3. If the soldier has enough money, the amount to borrow is 0.
 * 4. Otherwise, the amount to borrow is the total cost minus the initial money.
 *
 * Time Complexity: O(1), as the solution uses a direct mathematical formula.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  long long k, s, n;
  cin >> k >> s >> n;

  // Functional Utility: Calculates the total cost for 'n' bananas using the sum of an arithmetic series.
  // The result (total cost) is stored back into the variable 'n'.
  n = (n * (n + 1) * k) / 2;

  /**
   * @brief This block determines the amount of money to borrow.
   * Pre-condition: 'n' now holds the total cost, and 's' holds the initial money.
   */
  if (s > n)
    // If the soldier's money 's' is greater than the total cost 'n', they need to borrow nothing.
    cout << "0";
  else
    // Otherwise, the deficit is the total cost minus the soldier's money.
    printf("%lld", n - s);
}
