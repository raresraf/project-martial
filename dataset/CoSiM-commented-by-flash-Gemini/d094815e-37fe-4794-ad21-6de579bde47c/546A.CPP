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

#include <bits/stdc++.h> // Includes iostream, etc.
using namespace std;

int main(int argc, char const *argv[]) {
  // Functional Utility: 'k' stores the cost of the first banana (base cost).
  int k;
  // Functional Utility: 'n' stores the amount of money the soldier initially has.
  int n;
  // Functional Utility: 'w' stores the number of bananas the soldier wants to buy.
  int w;

  // Block Logic: Reads the base cost 'k', initial money 'n', and number of bananas 'w' from standard input.
  cin >> k >> n >> w;

  // Functional Utility: 'price' accumulates the total cost of buying 'w' bananas.
  int price = 0;
  /**
   * Block Logic: Calculates the total cost of 'w' bananas.
   * The i-th banana costs i * k.
   * Invariant: In each iteration, the cost of the current banana (i * k) is added to 'price'.
   */
  for (int i = 1; i <= w; i++) {
    price += (k * i);
  }

  /**
   * Block Logic: Determines the amount of money to borrow.
   * Pre-condition: 'price' holds the total cost, and 'n' holds initial money.
   * Invariant: If total cost exceeds initial money, the difference is borrowed; otherwise, 0 is borrowed.
   */
  if (price - n > 0)
    // Functional Utility: Prints the amount to borrow.
    cout << price - n;
  else
    // Functional Utility: Prints 0 if no money needs to be borrowed.
    cout << "0";
  return 0;
}
