/**
 * @file 69A.CPP
 * @brief This program solves the "Panoramix's Prediction" problem (Codeforces 80A).
 *   It checks if a given prime number `num2` is the very next prime number after another
 *   given prime number `num1`. Both `num1` and `num2` are positive integers.
 *   Problem description: http://codeforces.com/problemset/problem/80/A
 * * Algorithm: Prime number checking and sequential search.
 *   1. A helper function `is_prime` determines if a given integer is prime.
 *   2. The main function first validates that both input numbers (`num1` and `num2`) are prime.
 *   3. If they are, it then sequentially searches for any prime number between `num1` and `num2`.
 *   4. If an intermediate prime is found, `num2` is not the *next* prime. Otherwise, it is.
 * Time Complexity: O((num2 - num1) * sqrt(num2)) in the worst case, due to repeated `is_prime` calls
 *   within the search loop. For typical competitive programming constraints (e.g., num1, num2 <= 1000), this is efficient enough.
 * Space Complexity: O(1) for a few integer variables.
 */

/*
    Problem link:
    http://codeforces.com/problemset/problem/80/A
*/

#include <bits/stdc++.h> // Includes most standard C++ libraries, including <iostream>, <cmath>.

using namespace std;

/**
 * @brief Checks if a given integer is a prime number.
 * Functional Utility: Determines primality using trial division up to the square root of the number.
 * @param num The integer to check for primality.
 * @return True if `num` is prime, False otherwise.
 */
bool is_prime(const int &num) {
  // Conditional Logic: Numbers less than 2 are not prime.
  if (num < 2)
    return false;

  /**
   * Block Logic: Iterates from 2 up to the square root of `num` (inclusive).
   * Pre-condition: `num` is an integer >= 2.
   * Invariant: If `num` is divisible by any `factor` in this range (and `num` itself is not `factor`),
   *            then `num` is not prime.
   */
  for (int factor = 2; factor * factor <= num; factor++)
    if (num % factor == 0 and num != factor) // If `num` is divisible by `factor` and `factor` is not `num` itself.
      return false;                          // `num` is not prime.

  return true; // If no divisors are found, `num` is prime.
}

/**
 * @brief Main function to determine if num2 is the next prime after num1.
 * Functional Utility: Reads two numbers, validates their primality, searches for
 *                     intermediate primes, and prints "YES" or "NO".
 */
int main() {
  // Functional Utility: Optimizes C++ standard I/O streams for faster execution.
  ios_base::sync_with_stdio(false);

  int num1, num2; // `num1`: the first prime, `num2`: the candidate for the next prime.
  // Block Logic: Reads the two input integers.
  cin >> num1 >> num2;

  /**
   * Conditional Logic: First, checks if both `num1` and `num2` are prime.
   * If either is not prime, `num2` cannot be the next prime after `num1`.
   */
  if (is_prime(num1) and is_prime(num2)) {
    bool found_prime = false; // Flag to indicate if a prime is found between `num1` and `num2`.
    /**
     * Block Logic: Searches for any prime number between `num1` and `num2`.
     * Pre-condition: `num1` and `num2` are prime, `num1 < num2`.
     * Invariant: `found_prime` becomes true if any `num` in the range is prime.
     */
    for (int num = num1 + 1; num < num2; num++)
      if (is_prime(num)) {
        found_prime = true; // An intermediate prime was found.
        break;              // No need to check further.
      }

    // Conditional Logic: Based on whether an intermediate prime was found.
    if (found_prime)
      cout << "NO\n"; // An intermediate prime exists, so `num2` is not the *next* prime.
    else
      cout << "YES\n"; // No intermediate prime, so `num2` is the next prime.
  } else
    cout << "NO\n"; // Either `num1` or `num2` (or both) are not prime.

  return 0;
}