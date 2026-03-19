/**
 * @file 898B.CPP
 * @brief Solution to Codeforces Problem 898B: "A. Phone Numbers".
 *
 * This problem asks whether it's possible to represent a given sum `c` using
 * non-negative integer multiples of two other given numbers `a` and `b`.
 * In mathematical terms, determine if there exist non-negative integers `x` and `y`
 * such that `a*x + b*y = c`.
 *
 * Algorithm: Iterative search.
 * The algorithm iterates through possible non-negative multiples of `a` (i.e., `a*x`)
 * such that `a*x` does not exceed `c`. For each `a*x`, it calculates the remaining
 * value `val = c - a*x`. If `val` is non-negative and divisible by `b`, then a solution
 * (`x`, `val/b`) is found, and "YES" is printed along with `x` and `val/b`.
 * If no such pair is found after checking all valid multiples of `a`, "NO" is printed.
 *
 * Time Complexity: O(C/A), where C is the target sum and A is the first number.
 *                  In the worst case, it iterates `c/a` times.
 * Space Complexity: O(1), as only a few variables are used.
 */
#include <bits/stdc++.h> // Includes various standard libraries.
#include <iostream>      // Specifically includes iostream for standard input/output.
using namespace std;     // Uses the standard namespace to avoid prefixing std::
#define lli long long int // Defines 'lli' as a shorthand for 'long long int' to handle large numbers.

/**
 * @brief Main function to solve the representation problem.
 * Reads the target sum `c`, and two numbers `a` and `b`, then determines
 * if `c` can be represented as `a*x + b*y` for non-negative integers `x, y`.
 *
 * @return int Returns 0 on successful execution.
 */
int main() {
  lli c; // Stores the target sum.
  lli a; // Stores the first number.
  lli b; // Stores the second number.
  cin >> c; // Reads the target sum.
  cin >> a; // Reads the first number.
  cin >> b; // Reads the second number.
  
  /**
   * Block Logic: Iterates through possible non-negative multiples of 'a'.
   * Pre-condition: 'c', 'a', 'b' are read.
   * Invariant: 'i' represents the current non-negative multiple of 'a'.
   */
  for (lli i = 0; a * i <= c; i++) {
    lli val = c - a * i; // Calculates the remaining value after subtracting a*i from c.
    // Block Logic: Checks if the remaining value is non-negative and perfectly divisible by 'b'.
    if (val % b == 0) {
      cout << "YES" << endl; // Prints "YES" if a solution is found.
      cout << i << " " << (val / b) << endl; // Prints the non-negative integers x and y.
      return 0; // Exits the program as a solution is found.
    }
  }
  cout << "NO" << endl; // If the loop completes without finding a solution, prints "NO".
  return 0; // Indicates successful program execution.
}