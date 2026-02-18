/**
 * @file 546A.CPP
 * @brief Solution to the "Soldier and Bananas" problem (546A) from Codeforces.
 * @details This program calculates the amount of money a soldier needs to borrow to buy a
 * specific number of bananas, where the price of each subsequent banana increases linearly.
 *
 * @algorithm The solution calculates the total cost of all bananas and then determines the deficit.
 * 1. Read the initial cost of one banana (k), the soldier's initial money (n), and the number of bananas to buy (w).
 * 2. Calculate the total cost iteratively. The i-th banana costs `i * k`. The loop sums these costs for all `w` bananas.
 *    This is equivalent to calculating the sum of an arithmetic series: k * (1 + 2 + ... + w).
 * 3. Subtract the soldier's initial money from the total cost to find the amount to borrow.
 * 4. If the result is positive, it's the amount needed. If it's zero or negative, the soldier has enough money, so 0 is output.
 *
 * @complexity
 * Time Complexity: O(W) - The program uses a loop that iterates 'w' times to calculate the total cost.
 * Space Complexity: O(1) - The program uses a fixed number of variables.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int n, k, w, cnt = 0;
  cin >> k >> n >> w;
  /**
   * @block
   * @brief Calculates the total cost of 'w' bananas.
   * @invariant 'cnt' accumulates the cost. The i-th banana costs i*k.
   * @pre 'k' is the base cost, 'w' is the number of bananas.
   * @post 'cnt' holds the total cost of 'w' bananas.
   */
  for (int i = 1; i <= w; i++) {
    cnt += (i * k);
  }
  // Calculate the amount the soldier needs to borrow.
  cnt -= n;
  /**
   * @block
   * @brief Outputs the amount to borrow.
   * @pre 'cnt' is the difference between total cost and initial money.
   * @post Prints the amount to borrow, which is 'cnt' if positive, or 0 otherwise.
   */
  if (cnt <= 0) {
    cout << 0;
  } else {
    cout << cnt;
  }
  return 0;
}
