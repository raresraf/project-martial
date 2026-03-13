/**
 * @file 69A.CPP
 * @brief Solution for CodeForces Problem 80A - "Panoramix's Prediction".
 * @see http://codeforces.com/problemset/problem/80/A
 *
 * @note The filename `69A.CPP` is misleading. The code solves problem 80A, not 69A.
 *
 * @details
 * The problem asks to determine if a given number `num2` is the next prime number
 * immediately following another given prime number `num1`.
 *
 * @algorithm
 * 1. Define a helper function `is_prime()` to check for primality.
 * 2. Read the two numbers, `num1` and `num2`.
 * 3. First, check if both `num1` and `num2` are prime numbers themselves. If not, the condition fails.
 * 4. If they are both prime, iterate through all integers between `num1` and `num2`.
 * 5. In the loop, if any other prime number is found, it means `num2` is not the *next* prime.
 *    Set a flag and break.
 * 6. If the loop completes without finding any intermediate primes, `num2` is the next prime after `num1`.
 * 7. Print "YES" or "NO" accordingly.
 *
 * @time_complexity O(sqrt(M) * (M-N)), where N and M are the two input numbers. The `is_prime`
 *                  check takes sqrt(k) and it is called for numbers between N and M.
 * @space_complexity O(1).
 */

#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Checks if a given integer is a prime number.
 * @param num The integer to check.
 * @return True if the number is prime, false otherwise.
 * @logic A number is prime if it's greater than 1 and not divisible by any
 *        integer from 2 up to its square root.
 */
bool is_prime(const int &num) {
  if (num < 2)
    return false;

  for (int factor = 2; factor * factor <= num; factor++)
    if (num % factor == 0) // The condition `and num != factor` is redundant here.
      return false;

  return true;
}

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(nullptr);

  int num1, num2;
  cin >> num1 >> num2;

  // Pre-condition: Both numbers must be prime for num2 to be the next prime after num1.
  if (is_prime(num1) and is_prime(num2)) {
    bool found_intermediate_prime = false;
    /**
     * @brief Loop to check for any prime numbers between num1 and num2.
     * Invariant: `found_intermediate_prime` is false if no primes have been found
     * in the range (num1, current_num).
     */
    for (int num = num1 + 1; num < num2; num++) {
      if (is_prime(num)) {
        found_intermediate_prime = true;
        break; // Found a prime in between, so num2 cannot be the next prime.
      }
    }

    if (found_intermediate_prime)
      cout << "NO
";
    else
      cout << "YES
";
  } else {
    // If either num1 or num2 is not prime, the condition is not met.
    cout << "NO
";
  }

  return 0;
}
