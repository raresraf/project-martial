/**
 * @file 546A.CPP
 * @brief This program calculates the amount of money a soldier needs to borrow from a friend
 * to buy 'w' bananas. The first banana costs 'k' dollars, the second '2k', and so on.
 * The soldier initially has 'n' dollars.
 *
 * Algorithm: Iterative summation to calculate total cost, then conditional subtraction for borrowed amount.
 * Time Complexity: O(W) where W is the number of bananas.
 * Space Complexity: O(1)
 */

#include <cstdio>  // Unused, as iostream is used for input/output.
#include <iostream>
#include <string.h> // Unused, as no string manipulation is performed.
using namespace std;

// Note: 'main' without return type is a C-style feature, not standard C++.
main() {
  // Functional Utility: 'm', 'f1', 'f2' are declared but unused in this code.
  int m, k, i, w, f1, f2, p = 0, x;
  // Functional Utility: 'k' stores the cost of the first banana (base cost).
  // Functional Utility: 'i' is a loop counter.
  // Functional Utility: 'w' stores the number of bananas the soldier wants to buy.
  // Functional Utility: 'p' accumulates the total cost of buying 'w' bananas.
  // Functional Utility: 'x' stores the amount of money to be borrowed.

  // Functional Utility: 'n' stores the amount of money the soldier initially has.
  long int n;

  // Block Logic: Reads the base cost 'k', initial money 'n', and number of bananas 'w' from standard input.
  cin >> k >> n >> w;

  /**
   * Block Logic: Calculates the total cost of 'w' bananas.
   * The i-th banana costs i * k.
   * Invariant: In each iteration, the cost of the current banana (i * k) is added to 'p'.
   */
  for (i = 1; i <= w; i++) {
    p = p + k * i;
  }
  // Functional Utility: Calculates the potential amount to borrow (total cost minus initial money).
  x = p - n;

  /**
   * Block Logic: Determines the actual amount of money to borrow.
   * Pre-condition: 'x' holds the potential borrowed amount.
   * Invariant: If 'x' is positive, that amount is borrowed; otherwise, 0 is borrowed.
   */
  if (x <= 0) {
    // Functional Utility: Prints 0 if no money needs to be borrowed.
    cout << 0;
  } else {
    // Functional Utility: Prints the amount to borrow.
    cout << x;
  }
}
