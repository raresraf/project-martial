/**
 * @file 546A.CPP
 * @brief Solution to a problem calculating the money a soldier needs to borrow
 *        to buy bananas with increasing cost.
 *
 * This program takes three inputs: the cost of the first banana (k),
 * the initial money the soldier has (n), and the number of bananas to buy (w).
 * The cost of each subsequent banana increases by 'k' (1st: k, 2nd: 2k, ..., w-th: wk).
 * It calculates the total cost of 'w' bananas by iteratively summing their individual costs.
 * Then, it determines the amount of money the soldier needs to borrow,
 * outputting 0 if no borrowing is necessary (i.e., if initial money is sufficient or surplus).
 *
 * Algorithm: Iterative summation to calculate total cost, followed by conditional subtraction.
 * Time Complexity: O(W) where W is the number of bananas, as the loop runs W times.
 * Space Complexity: O(1) for variable storage.
 */
#include <iostream>
using namespace std;
typedef long long ll; // Type definition for 'long long' as 'll' for convenience.

int main() {
  // Declare variables for the cost of the first banana 'k',
  // initial money 'n', and number of bananas 'w'. Using 'long long' to handle large values.
  ll k, n, w;
  // Read the input values for k, n, and w.
  cin >> k >> n >> w;
  // Declare 'dollar' (total cost) and initialize to 0. Using 'long long' to accumulate large sums.
  ll dollar = 0;
  /**
   * Block Logic: Iterates from 1 to 'w' to calculate the total cost of all bananas.
   * Invariant: In each iteration, 'dollar' accumulates the cost of the current banana (k * i).
   */
  for (int i = 1; i <= w; i++) {
    // Add the cost of the i-th banana (i * k) to the total sum 'dollar'.
    dollar += k * i;
  }
  /**
   * Block Logic: Determines the amount of money to be borrowed or outputs "0" if sufficient funds exist.
   * Pre-condition: 'dollar' is the total cost of bananas, 'n' is the initial money.
   */
  if ((dollar - n) < 0) {
    // If the difference (total cost - initial money) is negative, it means no borrowing is needed.
    cout << "0";
  } else
    // Otherwise, output the positive difference, which is the amount to be borrowed.
    cout << dollar - n;
  return 0; // Indicate successful program execution.
}