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
#include <iostream>
using namespace std;

int main() {
  // Declare variables for the cost of the first banana 'k',
  // initial money 'n', number of bananas 'w'.
  // 'counter' tracks the current multiplier for 'k', starting at 1.
  // 'sum' accumulates the total cost, initialized to 0.
  int n, k, w, counter = 1, sum = 0;
  // Read the input values for k, n, and w.
  cin >> k >> n >> w;
  /**
   * Block Logic: Iterates 'w' times to calculate the total cost of all bananas.
   * Invariant: In each iteration, 'sum' accumulates the cost of the current banana (k * counter).
   * Note: The variable 'i' is declared but not used inside the loop body,
   *       'counter' effectively serves its purpose as the multiplier.
   */
  for (int i = 1; i <= w; i++) {
    // Add the cost of the current banana (k multiplied by the current 'counter' value) to the total sum.
    sum += k * counter;
    // Increment the 'counter' for the next banana's cost calculation.
    counter++;
  }
  /**
   * Block Logic: Determines the amount of money to be borrowed or outputs "0" if sufficient funds exist.
   * Pre-condition: 'sum' is the total cost of bananas, 'n' is the initial money.
   */
  if (sum > n)
    // If the total cost 'sum' is greater than the initial money 'n', output the deficit.
    cout << sum - n << "\n";
  else
    // Otherwise, output "0" as no borrowing is needed.
    cout << "0\n";

  return 0; // Indicate successful program execution.
}
