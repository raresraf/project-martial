/**
 * @5fec86c2-2465-4c40-8b2e-fb5623cf3655/546A.CPP
 * @brief This program calculates the amount of money a person needs to borrow.
 * Given 'w' items to buy, where the i-th item costs 'i * k' (or 'i * n' in this code's variable mapping) dollars,
 * and the person has 'n' (or 'k' in this code's variable mapping) dollars, it determines the deficit.
 * The total cost is calculated using the sum of an arithmetic progression.
 *
 * Algorithm: Arithmetic series summation, subtraction, and conditional output.
 * Time Complexity: O(1) per test case, as it involves a constant number of arithmetic operations.
 * Space Complexity: O(1) for storing variables.
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes iostream, cmath (for abs).
using namespace std;     // Uses the standard namespace.
#define ll long long // Defines 'll' as a shorthand for long long to handle potentially large values.

int main() {
  ll n, m, i, j, ans, k, w; // Declares variables: n (cost multiplier), m (sum 1 to w),
                            // i (total cost), j (unused), ans (amount to borrow), k (money owned), w (number of items).
  /**
   * Block Logic: Processes multiple test cases. Reads three integers (n, k, w)
   * for each test case until End-Of-File (EOF) is reached.
   * Invariant: For each iteration, the program calculates and prints the borrowing amount for the given inputs.
   */
  while (cin >> n >> k >> w) { // Reads n, k, w from standard input.
    // Block Logic: Calculates the sum of the series 1 + 2 + ... + w.
    // This is the sum of coefficients for the cost calculation.
    m = (w * (w + 1)) / 2; 
    // Inline: Calculates the total cost of buying 'w' items.
    // Each item 'x' costs 'n * x', so the total is 'n * (1 + 2 + ... + w)'.
    i = n * m;             
    
    /**
     * Block Logic: Determines the amount of money to borrow.
     * Pre-condition: 'i' is the total cost and 'k' is the available money.
     * Invariant: Prints the positive difference (borrowed amount) or "0" if no borrowing is needed.
     */
    if (i > k) { // If the total cost 'i' exceeds the available money 'k'.
      ans = abs(i - k); // Inline: Calculates the absolute difference, which is the amount to borrow.
      cout << ans << endl; // Prints the amount to borrow, followed by a newline.
    } else {
      cout << "0" << endl; // Prints "0" if available money is sufficient or more.
    }
  }
}
