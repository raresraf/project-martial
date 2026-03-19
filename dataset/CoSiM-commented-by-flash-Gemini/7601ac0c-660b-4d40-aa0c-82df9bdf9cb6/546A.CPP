/**
 * @file 546A.CPP
 * @brief Solution to Codeforces Problem 546A: "Soldier and Bananas".
 *
 * This problem involves a soldier who wants to buy `w` bananas. The first banana
 * costs `k` dollars, the second `2k` dollars, and so on, with the `i`-th banana
 * costing `i * k` dollars. The soldier has `n` dollars. The task is to calculate
 * how much money the soldier needs to borrow to buy all `w` bananas.
 *
 * Algorithm: Arithmetic series sum.
 * The total cost of `w` bananas is the sum of an arithmetic series:
 * k + 2k + 3k + ... + wk = k * (1 + 2 + 3 + ... + w) = k * w * (w + 1) / 2.
 * The algorithm calculates this total cost and then subtracts the money the soldier has (`n`).
 * If the result is negative or zero, no money needs to be borrowed, so 0 is printed.
 * Otherwise, the positive difference is printed.
 *
 * Time Complexity: O(1), as it involves a constant number of arithmetic operations.
 * Space Complexity: O(1), as only a few variables are used.
 */
#include <bits/stdc++.h> // Includes various standard libraries, such as iostream for input/output.
using namespace std;     // Uses the standard namespace to avoid prefixing std::
typedef long long ll;    // Defines 'll' as a shorthand for 'long long' to handle potentially large sums.

/**
 * @brief Main function to solve the "Soldier and Bananas" problem.
 * Reads the cost of the first banana, the initial money, and the number of bananas,
 * then calculates and prints the amount of money the soldier needs to borrow.
 *
 * @return int Returns 0 on successful execution.
 */
int main() {
  int k; // Cost of the first banana.
  int n; // Initial money the soldier has.
  int w; // Number of bananas the soldier wants to buy.
  cin >> k >> n >> w; // Reads k, n, and w from standard input.
  
  // Block Logic: Calculates the total cost of all 'w' bananas using the arithmetic series sum formula.
  // The 'll' cast ensures that the intermediate product w * (w + 1) / 2 * k does not overflow if k and w are large.
  ll total = (ll)w * (w + 1) / 2 * k;
  
  /**
   * Block Logic: Determines the amount of money to borrow.
   * Pre-condition: 'total' is the calculated cost, 'n' is the money on hand.
   * Invariant: If 'total' is less than or equal to 'n', no money needs to be borrowed.
   * Otherwise, the difference is the amount to borrow.
   */
  if (total <= n) {
    cout << 0; // Prints 0 if no money needs to be borrowed.
  } else {
    cout << total - n; // Prints the amount of money to borrow.
  }
  return 0; // Indicates successful program execution.
}