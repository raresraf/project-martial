/**
 * @file 546A.CPP
 * @brief Solution to a problem calculating the money a soldier needs to borrow
 *        to buy bananas with increasing cost, using an optimized formula.
 *
 * This program takes three integers: the cost of the first banana (k),
 * the initial money the soldier has (n), and the number of bananas to buy (w).
 * The cost of each subsequent banana increases by 'k' (1st: k, 2nd: 2k, ..., w-th: wk).
 * It calculates the total cost of 'w' bananas using the sum of an arithmetic progression.
 * Then, it determines the amount of money the soldier needs to borrow,
 * outputting 0 if no borrowing is necessary (i.e., if initial money is sufficient).
 *
 * Algorithm: Direct application of the arithmetic series sum formula:
 *            Total Cost = k * (w * (w + 1)) / 2. Followed by conditional subtraction.
 * Time Complexity: O(1) as it involves a constant number of arithmetic operations.
 * Space Complexity: O(1) for variable storage.
 */
#include <iostream>
#include <stdio.h> // Included but not explicitly used for I/O in this version.
using namespace ::std; // Using the standard namespace.

// Note: The 'main' function signature without 'int' return type is a compiler extension.
main() {
  // Functional Utility: Disables synchronization between C-style I/O and C++ streams,
  // speeding up input/output operations, common in competitive programming.
  ios_base::sync_with_stdio(false); // Fast I/O
  // Declare variables for the cost of the first banana 'k',
  // initial money 'n', and number of bananas 'w'.
  int k, n, w;
  // Read the input values for k, n, and w.
  cin >> k >> n >> w;
  // Calculate the total cost 't' using the arithmetic series sum formula.
  // Using 'long int' for 't' to prevent potential overflow for large k or w.
  long int t = k * (w * (w + 1)) / 2;
  /**
   * Block Logic: Determines the amount of money to be borrowed or outputs 0 if sufficient funds exist.
   * Pre-condition: 'n' is the initial money, 't' is the total cost of bananas.
   */
  if (n >= t)
    // If initial money 'n' is greater than or equal to total cost 't', no borrowing is needed.
    cout << 0;
  else
    // Otherwise, calculate and output the amount that needs to be borrowed.
    cout << t - n;
  return 0; // Indicate successful program execution.
}