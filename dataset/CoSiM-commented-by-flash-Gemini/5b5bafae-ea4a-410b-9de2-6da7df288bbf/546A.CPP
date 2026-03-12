/**
 * @5b5bafae-ea4a-410b-9de2-6da7df288bbf/546A.CPP
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
  cin >> k >> n >> w; // Reads 'k', 'n', and 'w' from standard input.

  int total = 0; // Initializes 'total' cost to 0.
  
  /**
   * Block Logic: Calculates the total cost of buying 'w' bananas.
   * The cost of the i-th banana is `i * k`.
   * Invariant: 'total' accurately accumulates the total cost after each banana is accounted for.
   */
  for (int i = 1; i <= w; i++) { // Loop from the 1st banana to the w-th banana.
    total += (k * i);            // Inline: Calculates the cost of the current banana (`i * k`) and adds it to `total`.
  }

  /**
   * Block Logic: Determines the amount of money to borrow and prints the result.
   * Pre-condition: 'total' contains the total cost of bananas and 'n' is the available money.
   * Invariant: Prints "0" if available money is sufficient, otherwise prints the amount to borrow.
   */
  // Inline: Uses a ternary operator to conditionally print "0" if `n` is greater than `total` (sufficient money),
  // otherwise prints the difference (`total - n`), which is the amount to borrow.
  n > total ? cout << "0" : cout << (total - n); 
  cout << endl; // Prints a newline character.

  return 0; // Indicates successful program termination.
}
