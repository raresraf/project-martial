/**
 * @file 898B.CPP
 * @brief Solves a linear Diophantine equation of the form ax + by = c.
 *
 * This program determines if there exist non-negative integers x and y
 * that satisfy the equation `a*x + b*y = c`, where `c`, `a`, and `b` are
 * given positive integers. If a solution exists, it prints "YES" followed
 * by the values of x and y. Otherwise, it prints "NO".
 *
 * @algorithm
 * The problem is to find a pair of non-negative integers (x, y) for the equation.
 * This can be solved by iterating through all possible values for one variable
 * and checking if the resulting equation can be satisfied by the other variable.
 *
 * 1. Read the target value `c` and the coefficients `a` and `b`.
 * 2. Iterate through all possible non-negative integer values for `x` (named `i` in the code).
 *    The loop runs as long as `a * i` is less than or equal to `c`, because if `a * i > c`,
 *    then `b * y` would have to be negative, which is not allowed.
 * 3. Inside the loop, for each value of `i` (representing `x`):
 *    a. Calculate the remaining value needed: `val = c - a * i`. This is the value
 *       that `b * y` must equal.
 *    b. Check if `val` is a non-negative multiple of `b`. Since `val` will always be
 *       non-negative due to the loop condition, we only need to check if `val` is
 *       perfectly divisible by `b`. This is done using the modulo operator: `val % b == 0`.
 * 4. If a valid `val` is found:
 *    a. A solution exists. Print "YES".
 *    b. Print `i` (the value of `x`) and `val / b` (the value of `y`).
 *    c. Terminate the program, as we only need to find one solution.
 * 5. If the loop completes without finding any solution, it means no pair of non-negative
 *    integers (x, y) satisfies the equation. Print "NO".
 *
 * @complexity
 * Time Complexity: O(c/a), as the loop iterates from `i=0` up to `c/a`.
 * Space Complexity: O(1).
 */

#include <bits/stdc++.h> // Includes most standard libraries, common in competitive programming.
#include <iostream>
using namespace std;

// Define a type alias for long long int for convenience.
#define lli long long int

int main() {
  // c: the target sum
  // a, b: the coefficients
  lli c, a, b;
  cin >> c;
  cin >> a;
  cin >> b;

  /**
   * @brief Brute-force search for the 'x' coefficient (named 'i').
   *
   * The loop iterates through possible values of the first number of items,
   * checking if the remainder can be formed by the second number of items.
   */
  for (lli i = 0; a * i <= c; i++) {
    // Calculate the remaining value after using `i` items of cost `a`.
    lli val = c - a * i; // Use lli to avoid overflow if c, a, i are large.
    
    // Check if the remaining value is perfectly divisible by `b`.
    if (val % b == 0) {
      // If it is, a solution is found.
      cout << "YES" << endl;
      // The number of items of cost `a` is `i`.
      // The number of items of cost `b` is `val / b`.
      cout << i << " " << (val / b) << endl;
      return 0; // Exit after finding the first solution.
    }
  }

  // If the loop finishes, no solution was found.
  cout << "NO" << endl;
  return 0;
}
