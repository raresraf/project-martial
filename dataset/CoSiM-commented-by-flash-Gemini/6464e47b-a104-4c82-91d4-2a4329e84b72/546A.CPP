/**
 * @file 546A.CPP
 * @brief Solution for Codeforces Problem 546A - "Soldier and Bananas".
 * This program calculates the amount of money a soldier needs to borrow to buy a certain
 * number of bananas, given the cost progression and the money the soldier already has.
 * The cost of the i-th banana is i*k.
 * Algorithm: Direct arithmetic calculation of the sum of an arithmetic progression.
 * Time Complexity: O(1) - Involves a fixed number of arithmetic operations.
 * Space Complexity: O(1) - Uses a constant amount of memory for variables.
 */
#include <iostream> // Required for standard input/output operations (cin, cout)
using namespace std; // Uses the standard namespace to avoid prefixing std::

int main() {
  // k: the cost of the first banana.
  // n: the amount of money the soldier has.
  // w: the number of bananas the soldier wants to buy.
  // i, j: unused variables, possibly remnants from a template or previous implementation.
  // sum: stores the total calculated cost for 'w' bananas.
  long long int k, n, w, i, j, sum = 0;
  
  // Reads the input values: k (first banana cost), n (money soldier has), w (number of bananas).
  cin >> k >> n >> w;
  
  /**
   * Block Logic: Calculates the total cost of 'w' bananas.
   * The cost forms an arithmetic progression: k, 2k, 3k, ..., wk.
   * The sum of this series is k * (1 + 2 + ... + w) = k * w * (w + 1) / 2.
   * Invariant: 'sum' holds the total money required to buy 'w' bananas.
   */
  sum = (k * w * (w + 1)) / 2;
  
  // Calculates the amount of money the soldier needs to borrow.
  // If this value is negative, it means the soldier has enough or more money.
  long long int ans = sum - n;
  
  // Conditional output: If 'ans' is negative (soldier has enough money), output 0.
  // Otherwise, output the calculated amount to borrow.
  if (ans < 0)
    cout << "0";
  else
    cout << ans;
    
  return 0; // Indicates successful program execution.
}

