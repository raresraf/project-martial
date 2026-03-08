/**
 * @file 546A.CPP
 * @brief Solution for the "Soldier and Bananas" problem.
 *
 * @details
 * This program calculates the amount of money a soldier needs to borrow to buy
 * a certain number of bananas. The cost of the bananas increases arithmetically.
 * The program first calculates the sum of the cost multipliers (1 + 2 + ... + w)
 * using a loop, then determines the total cost, and finally computes the deficit
 * based on the money the soldier initially has. The logic is encapsulated
 * within a `solve` function. The original source contained Vietnamese comments.
 *
 * @section
 * Algorithm: Two-stage calculation involving iterative summation.
 * Time Complexity: O(W), where W is the number of bananas, due to the loop.
 * Space Complexity: O(1).
 *
 * @note An O(1) solution is possible by replacing the summation loop with the
 * direct formula for an arithmetic series sum: `w * (w + 1) / 2`.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Contains the core logic to solve the problem.
 */
void solve() {
  int k, w;    // k: cost of the first banana, w: number of bananas
  long long n; // n: the amount of money the soldier has
  cin >> k >> n >> w;

  // 's' will accumulate the sum of multipliers (1 + 2 + ... + w).
  int s = 0;
  /**
   * @brief This loop calculates the sum of the first 'w' integers.
   */
  for (int i = 1; i <= w; i++) {
    s += i;
  }

  // 'kq' (result) stores the final amount to borrow.
  // The total cost is (k * s), and we subtract the money on hand 'n'.
  // Casting to long long prevents potential overflow during multiplication.
  long long kq = (long long)k * s - n;

  /**
   * @brief Print the result.
   *
   * Logic: If the calculated deficit 'kq' is positive, that's the amount
   * to borrow. Otherwise, the soldier has enough money, so they borrow 0.
   */
  if (kq > 0)
    cout << kq;
  else
    cout << 0;
}

/**
 * @brief The main function, which serves as the entry point.
 */
int main() {

  // Call the function containing the problem's solution.
  solve();
  return 0;
}
