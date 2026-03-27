/**
 * @file 546A.CPP
 * @brief Calculates the amount of money a person needs to borrow to buy 'n' items,
 *        given that the first item costs 'k', the second '2k', and so on,
 *        and the person initially has 's' money.
 *
 * Algorithm:
 * 1. Read the base cost (k), initial money (s), and number of items (n).
 * 2. Calculate the total cost of 'n' items using the sum of an arithmetic progression:
 *    Total Cost = k * (1 + 2 + ... + n) = k * (n * (n + 1) / 2).
 * 3. Determine the amount to borrow by subtracting the initial money (s) from the total cost.
 * 4. If the result is negative or zero (meaning enough or more money is available), output 0.
 *    Otherwise, output the calculated borrowing amount.
 *
 * Time Complexity: O(1) - The operations are constant time arithmetic calculations.
 * Space Complexity: O(1) - Only a few integer variables are used.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Entry point of the program.
 * Functional Utility: Orchestrates the input of problem parameters, calculates the total cost,
 * and determines the required borrowing amount.
 * @return 0 on successful execution.
 */
int main() {
  // Block Logic: Declares variables to store the base cost (k),
  //              the initial money (s), and the number of items (n).
  long long k, s, n;
  // Block Logic: Reads the integer values for k, s, and n from standard input.
  cin >> k >> s >> n;
  /**
   * Block Logic: Calculates the total cost of 'n' items.
   * The cost forms an arithmetic progression: k, 2k, 3k, ..., nk.
   * The sum of this series is k * (1 + 2 + ... + n).
   * Inline: The sum of the first 'n' natural numbers is given by the formula n * (n + 1) / 2.
   * Functional Utility: Computes the cumulative cost for purchasing all desired items.
   */
  n = (n * (n + 1) * k) / 2;

  /**
   * Block Logic: Determines the amount of money the person needs to borrow.
   * If the total cost exceeds the initial money, the difference is the amount to borrow.
   * Otherwise, if the initial money is sufficient or more, no money needs to be borrowed.
   *
   * Pre-condition: `n` holds the calculated aggregate cost of items, and `s` is the available money.
   * Invariant: The output will be a non-negative integer representing the exact borrowing amount or zero.
   */
  if (s > n)
    cout << "0";
  else
    printf("%lld", n - s);
}
