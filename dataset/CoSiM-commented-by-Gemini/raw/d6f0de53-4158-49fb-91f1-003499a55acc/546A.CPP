/**
 * @file 546A.CPP
 * @brief Solution for Codeforces Problem 546A, "Soldier and Bananas".
 * @details This program calculates the amount of money a soldier needs to borrow
 * to buy a certain number of bananas, where the cost of each subsequent banana increases linearly.
 *
 * @algorithm Iterative Summation
 * @complexity
 * Time: O(W), where W is the number of bananas.
 * Space: O(1).
 */
#include <iostream>
using namespace std;

// Use a typedef for long long for convenience, a common practice in competitive programming.
typedef long long ll;

int main() {
  ll k, n, w; // k: cost of first banana, n: initial money, w: bananas to buy
  cin >> k >> n >> w;
  
  ll dollar = 0; // Use long long to prevent overflow.

  /**
   * @brief Loop to calculate the total cost for 'w' bananas.
   * The cost increases by 'k' for each banana.
   */
  for (int i = 1; i <= w; i++) {
    dollar += k * i;
  }

  /**
   * @brief Determine the amount of money to borrow.
   * If the total cost ('dollar') is less than or equal to the soldier's
   * initial money ('n'), the result is 0. Otherwise, it's the difference.
   */
  if ((dollar - n) <= 0) { // A more explicit check would be `if (dollar <= n)`
    cout << "0";
  } else {
    cout << dollar - n;
  }
  
  return 0; // Indicate successful execution.
}