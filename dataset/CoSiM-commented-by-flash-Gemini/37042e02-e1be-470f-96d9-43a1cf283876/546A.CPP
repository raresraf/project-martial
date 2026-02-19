/**
 * @file 546A.CPP
 * @brief Solution to a competitive programming problem: "Soldier and Bananas".
 *
 * This program calculates the amount of money a soldier needs to borrow to buy 'w' bananas.
 * The cost of the first banana is 'k', the second is '2k', and so on, up to 'wk'.
 * The soldier has 'n' dollars initially.
 *
 * Algorithm: Summation of an arithmetic progression to find total cost, then calculate deficit.
 * Time Complexity: O(w), where 'w' is the number of bananas, due to the loop for summing costs.
 *                  (Can be O(1) using the arithmetic series sum formula: k * w * (w + 1) / 2).
 */
#include <algorithm>     // Functional Utility: Included for standard algorithms (not strictly used here).
#include <cmath>         // Functional Utility: Included for mathematical functions (not strictly used here).
#include <iostream>      // Functional Utility: Included for standard input/output operations.
#include <string>        // Functional Utility: Included for string manipulation (not strictly used here).
#include <unordered_set> // Functional Utility: Included for unordered_set container (not strictly used here).
#include <vector>        // Functional Utility: Included for vector container (not strictly used here).

using namespace std; // Functional Utility: Uses the standard namespace to avoid prefixing standard library elements with std::

int main() {
  // Functional Utility: Optimizes standard I/O streams for faster execution,
  // crucial in competitive programming for time-sensitive tasks.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  // Variable Declarations:
  // k: Cost of the first banana.
  // n: Initial amount of money the soldier has.
  // w: Number of bananas the soldier wants to buy.
  // total_cost: Accumulator for the total cost of all bananas.
  // next_banana: (Declared but unused)
  int k, n, w;
  int total_cost = 0;
  int next_banana; // Functional Utility: This variable is declared but not used in the current logic.

  // Input: Reads the cost 'k', initial money 'n', and number of bananas 'w'.
  cin >> k >> n >> w;

  // Block Logic: Calculates the total cost of 'w' bananas.
  // The cost of the i-th banana is k * i.
  // Invariant: 'total_cost' accumulates the sum k*1 + k*2 + ... + k*i.
  for (int i = 1; i <= w; ++i) {
    total_cost += k * i; // Inline: Adds the cost of the i-th banana to the total.
  }

  // Block Logic: Determines the amount of money to borrow.
  // Precondition: 'total_cost' holds the final price of all bananas, 'n' is money on hand.
  if (n >= total_cost) { // If the soldier has enough money or more.
    cout << 0;           // Output: Prints 0, as no money needs to be borrowed.
  } else {               // If the soldier does not have enough money.
    cout << total_cost - n; // Output: Prints the amount of money that needs to be borrowed.
  }

  return 0; // Functional Utility: Indicates successful program execution.
}
