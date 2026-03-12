/**
 * @5abf1aa9-dc36-4877-a5d2-e5a366edbc0e/546A.CPP
 * @brief This program calculates the amount of money a soldier needs to borrow.
 * The cost of buying bananas increases linearly: the first costs 'k', the second '2k',
 * and so on, with the 'i'-th banana costing 'i * k' dollars. Given 'w' bananas to buy
 * and 'n' dollars the soldier already has, the program determines the deficit.
 *
 * Algorithm: Iterative summation to calculate total cost, followed by a subtraction and conditional output.
 * Time Complexity: O(W) where W is the number of bananas.
 * Space Complexity: O(1) for storing variables.
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes iostream.
using namespace std;     // Uses the standard namespace.

int main() {
  int k, n, w; // Declares 'k' (cost of first banana), 'n' (initial money), and 'w' (number of bananas).
  int total = 0; // Initializes 'total' cost to 0.
  int sol;       // Variable to store the solution (amount to borrow).

  cin >> k >> n >> w; // Reads 'k', 'n', and 'w' from standard input.
  
  /**
   * Block Logic: Calculates the total cost of buying 'w' bananas.
   * The cost of the i-th banana is `i * k`.
   * Invariant: 'total' accurately accumulates the total cost after each banana is accounted for.
   */
  for (int i = 1; i <= w; i++) { // Loop from the 1st banana to the w-th banana.
    total += k * i;              // Inline: Calculates the cost of the current banana (`i * k`) and adds it to `total`.
  }
  
  /**
   * Block Logic: Determines the amount of money to borrow.
   * Pre-condition: 'total' contains the total cost of bananas and 'n' is the available money.
   * Invariant: Prints the required borrowing amount or "0" if no borrowing is needed.
   */
  if (total < n) { // If the total cost is less than the available money.
    cout << "0" << endl; // Prints "0" as no borrowing is needed.
  } else { // If the total cost is greater than or equal to the available money.
    sol = total - n; // Inline: Calculates the amount to borrow.
    cout << sol << endl; // Prints the amount to borrow.
  }
  return 0; // Indicates successful program termination.
}