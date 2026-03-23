/**
 * @file 546A.CPP
 * @brief Solution for Codeforces Problem 546A, "Soldier and Bananas".
 * @details A soldier wants to buy 'w' bananas. The first banana costs 'k', the second costs '2k',
 * and so on. The soldier has 'n' dollars. This program calculates the amount of money
 * the soldier must borrow to buy all the bananas.
 *
 * @algorithm Arithmetic Series Summation
 * @complexity
 * Time: O(W), where W is the number of bananas. The loop runs W times to calculate the total cost.
 *       This could be optimized to O(1) by using the formula for the sum of an arithmetic series:
 *       Total Cost = k * W * (W + 1) / 2.
 * Space: O(1), using a fixed amount of memory.
 */
#include <bits/stdc++.h> // Includes all standard libraries
using namespace std;

// Typedefs for common data types in competitive programming
typedef long long ll;
typedef pair<int, int> ii;

int main() {
  // Fast I/O operations, common in competitive programming.
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);

  int k, n, w;
  long long ret = 0; // Use long long for total cost to prevent overflow.
  
  // k: cost of the first banana
  // n: initial amount of money
  // w: number of bananas to buy
  cin >> k >> n >> w;

  /**
   * @brief Calculate the total cost of 'w' bananas.
   * The cost of the i-th banana is i*k. The loop sums the cost for all bananas
   * from 1 to w.
   */
  for (int i = 1; i <= w; i++) {
    ret += i * k;
  }

  /**
   * @brief Calculate and print the amount to borrow.
   * This is the difference between the total cost and the money the soldier has.
   * If the soldier has enough money, the amount to borrow is 0.
   */
  cout << max(0LL, ret - n) << endl;

  return 0; // Indicate successful execution.
}