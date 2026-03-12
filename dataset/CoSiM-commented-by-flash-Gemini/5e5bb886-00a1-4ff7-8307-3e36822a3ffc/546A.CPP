/**
 * @5e5bb886-00a1-4ff7-8307-3e36822a3ffc/546A.CPP
 * @brief This program calculates the amount of money a soldier needs to borrow.
 * The cost of buying bananas increases linearly: the first costs 'k', the second '2k',
 * and so on, with the 'i'-th banana costing 'i * k' dollars. Given 'w' bananas to buy
 * and 'n' dollars the soldier already has, the program determines the deficit.
 *
 * Algorithm: Iterative summation to calculate total cost, followed by subtraction and conditional output.
 * Time Complexity: O(W) where W is the number of bananas.
 * Space Complexity: O(1) for storing variables.
 */

#include <iostream> // For standard input/output operations (cin, cout).
#include <queue>    // Included but not explicitly used in this snippet.
#include <stack>    // Included but not explicitly used in this snippet.
#include <string>   // Included but not explicitly used in this snippet.
#include <vector>   // Included but not explicitly used in this snippet.
using namespace std; // Uses the standard namespace.

int main() {
  long long k, n, w, sum = 0; // Declares 'k' (cost of first banana), 'n' (initial money),
                              // 'w' (number of bananas), and 'sum' (total cost) initialized to 0.
                              // All declared as long long to prevent overflow.
  cin >> k >> n >> w; // Reads 'k', 'n', and 'w' from standard input.
  
  /**
   * Block Logic: Calculates the total cost of buying 'w' bananas.
   * The cost of the i-th banana is `i * k`.
   * Invariant: 'sum' accurately accumulates the total cost after each banana is accounted for.
   */
  for (int i = 1; i <= w; i++) { // Loop from the 1st banana to the w-th banana.
    sum = sum + (long long)i * k; // Inline: Calculates the cost of the current banana (`i * k`) and adds it to `sum`.
                                  // (long long)i ensures multiplication is done with long long to prevent overflow.
  }
  
  // Inline: Calculates the net amount needed (or surplus) by subtracting available money 'n' from total cost 'sum'.
  sum = sum - n; 
  
  /**
   * Block Logic: Determines the amount of money to borrow and prints the result.
   * Pre-condition: 'sum' contains the net amount needed (can be negative if surplus).
   * Invariant: Prints "0" if available money was sufficient, otherwise prints the positive deficit.
   */
  if (sum <= 0) { // If the net amount needed is zero or negative (sufficient money).
    cout << 0 << endl; // Prints "0" as no borrowing is needed.
  } else { // If the net amount needed is positive (deficit).
    cout << sum << endl; // Prints the positive deficit (amount to borrow).
  }
  // The program implicitly returns 0 on successful completion of main.
}
