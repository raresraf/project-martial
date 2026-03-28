/**
 * @file 546A.CPP
 * @brief Solution to a problem calculating the money a soldier needs to borrow
 *        to buy bananas with increasing cost.
 *
 * This program takes three integers: the cost of the first banana (k),
 * the initial money the soldier has (n), and the number of bananas to buy (w).
 * The cost of each subsequent banana increases by 'k' (1st: k, 2nd: 2k, ..., w-th: wk).
 * It calculates the total cost of 'w' bananas by iteratively summing their individual costs.
 * Then, it determines the amount of money the soldier needs to borrow,
 * outputting 0 if no borrowing is necessary (i.e., if initial money is sufficient).
 *
 * Algorithm: Iterative summation to calculate total cost, followed by conditional subtraction.
 * Time Complexity: O(W) where W is the number of bananas, as the loop runs W times.
 * Space Complexity: O(1) for variable storage.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // Declare variables for the cost of the first banana 'k',
  // initial money 'n', number of bananas 'w', and 'sum' initialized to 0
  // to accumulate the total cost.
  int k, n, w, sum = 0;
  // Read the input values for k, n, and w.
  cin >> k >> n >> w;
  /**
   * Block Logic: Iterates from 1 to 'w' to calculate the total cost of all bananas.
   * Invariant: In each iteration, 'sum' accumulates the cost of the current banana (i * k).
   */
  for (int i = 1; i <= w; i++) {
    // Add the cost of the i-th banana (i * k) to the total sum.
    sum += k * i;
  }
  /**
   * Block Logic: Determines the amount of money to be borrowed or outputs 0 if sufficient funds exist.
   * Pre-condition: 'n' is the initial money, 'sum' is the total cost of bananas.
   */
  if (n >= sum) {
    // If initial money 'n' is greater than or equal to total cost 'sum', no borrowing is needed.
    cout << 0 << endl;
  } else {
    // Otherwise, calculate and output the amount that needs to be borrowed.
    cout << sum - n << endl;
  }

  return 0; // Indicate successful program execution.
}
