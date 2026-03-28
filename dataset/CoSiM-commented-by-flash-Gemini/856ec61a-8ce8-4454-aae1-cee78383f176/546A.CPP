/**
 * @file 546A.CPP
 * @brief Solution to a problem calculating the money a soldier needs to borrow
 *        to buy bananas with increasing cost.
 *
 * This program takes three integers: the cost of the first banana (k),
 * the initial money the soldier has (n), and the number of bananas to buy (w).
 * The cost of each subsequent banana increases by 'k' (1st: k, 2nd: 2k, ..., w-th: wk).
 * It calculates the total cost, then determines the amount of money the soldier
 * needs to borrow, outputting 0 if no borrowing is necessary.
 *
 * Algorithm: Iterative summation to calculate total cost, followed by conditional subtraction.
 * Time Complexity: O(W) where W is the number of bananas.
 * Space Complexity: O(1) for variable storage.
 */
#include <iostream>
using namespace std;

int main() {
  // Declare variables for the cost of the first banana 'k',
  // initial money 'n', and number of bananas 'w'.
  int k, n, w;
  // Read the input values for k, n, and w.
  cin >> k >> n >> w;

  // Initialize 'sum' to 0 to accumulate the total cost of bananas.
  int sum = 0;
  /**
   * Block Logic: Iterates from 1 to 'w' to calculate the total cost of all bananas.
   * Invariant: In each iteration, 'sum' accumulates the cost of the current banana (i * k).
   */
  for (int i = 1; i <= w; i++) {
    // Add the cost of the i-th banana (i * k) to the total sum.
    sum = sum + i * k;
  }
  // Calculate the amount to borrow: total cost minus initial money.
  int res = sum - n;
  /**
   * Block Logic: Determines whether borrowing is necessary and outputs the appropriate amount.
   * Pre-condition: 'res' holds the calculated deficit (or surplus).
   */
  if (res > 0)
    // If 'res' is positive, the soldier needs to borrow this amount.
    cout << res;
  else
    // If 'res' is zero or negative, no borrowing is needed.
    cout << 0;
  return 0;
}