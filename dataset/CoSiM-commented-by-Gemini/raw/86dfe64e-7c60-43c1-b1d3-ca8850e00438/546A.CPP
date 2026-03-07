/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem.
 *
 * @details
 * The program calculates the amount of money a soldier needs to borrow to buy a
 * certain number of bananas, where the cost of each banana increases linearly.
 *
 * Algorithm:
 * 1. Read the initial cost of one banana (`k`), the soldier's initial money (`n`),
 *    and the total number of bananas to buy (`w`).
 * 2. Calculate the total cost by iterating from 1 to `w` and summing up the
 *    cost of each banana (`i * k`).
 * 3. Subtract the soldier's money (`n`) from the total cost (`amt`).
 * 4. If this difference is positive, it represents the amount to borrow, which is printed.
 * 5. Otherwise, the soldier has enough money, and 0 is printed.
 *
 * Time Complexity: O(W), where W is the number of bananas, due to the iterative summation.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;

int main() {

  int k, n, w;
  int amt = 0; // To store the total cost of bananas.
  cin >> k >> n >> w;

  /**
   * @brief Loop to calculate the total cost of 'w' bananas.
   * The cost of the i-th banana is i * k. This loop sums the cost for all bananas.
   */
  for (int i = 1; i <= w; i++)
    amt += i * k;

  // Calculate the amount to borrow.
  if (amt - n > 0)
    // If the cost is greater than the money available, print the difference.
    cout << amt - n << endl;
  else
    // Otherwise, no money needs to be borrowed.
    cout << 0 << endl;
  return 0;
}
