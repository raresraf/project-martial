/**
 * @5bfd3418-68af-4fc3-83ad-ec80562f13d9/546A.CPP
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
  // Declares variables: k (cost of first banana), n (initial money), w (number of bananas),
  // i, j (loop counters - 'j' is unused), sum (total cost), a (amount to borrow).
  int k, n, w, i, j, sum = 0, a; 
  cin >> k >> n >> w; // Reads 'k', 'n', and 'w' from standard input.

  /**
   * Block Logic: Calculates the total cost of buying 'w' bananas.
   * The cost of the i-th banana is `i * k`.
   * Invariant: 'sum' accurately accumulates the total cost after each banana is accounted for.
   */
  for (i = 1; i <= w; i++) { // Loop from the 1st banana to the w-th banana.
    sum += (k * i);          // Inline: Calculates the cost of the current banana (`i * k`) and adds it to `sum`.
  }

  /**
   * Block Logic: Determines the amount of money to borrow and prints the result.
   * Pre-condition: 'sum' contains the total cost of bananas and 'n' is the available money.
   * Invariant: Prints the required borrowing amount or "0" if no borrowing is needed.
   */
  if (sum > n) // If the total cost exceeds the available money.
    a = sum - n; // Inline: Calculates the amount to borrow.
  else { // If the available money is sufficient or more.
    cout << "0" << endl; // Prints "0" as no borrowing is needed.
    return 0;            // Terminates the program early.
  }

  cout << a << endl; // Prints the amount to borrow, followed by a newline.
  // The program implicitly returns 0 here if `sum > n` was true.
}
