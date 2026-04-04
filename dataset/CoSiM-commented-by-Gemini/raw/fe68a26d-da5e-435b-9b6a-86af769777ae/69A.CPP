/**
 * @file 69A.CPP
 * @brief Determines if a number 'm' is the next prime number after 'n'.
 * @author Maciej Szeptuch (2013, II UWr)
 *
 * @details The program reads two integers, n and m. It then verifies two conditions:
 *          1. 'm' must be a prime number.
 *          2. There must be no prime numbers between n and m (exclusive of n, inclusive of m-1).
 *          If both conditions are met, it prints "YES", otherwise "NO".
 *
 * Algorithm:
 * 1. Read integers 'n' and 'm'.
 * 2. The main logic is in the `check(n + 1, m)` function call.
 * 3. `check(start, end)`:
 *    a. It first calls `isPrime(end)` to verify that 'm' is prime. If not, returns false.
 *    b. It then iterates from 'start' up to 'end - 1'.
 *    c. In the loop, it calls `isPrime` on the current number. If any number in this
 *       range is found to be prime, it means 'm' is not the *next* prime, so it returns false.
 *    d. If the loop finishes, it means no primes were found in the range (n, m), so it returns true.
 * 4. `isPrime(p)`:
 *    a. A basic trial division primality test. It checks for divisibility from 2 up to p-1.
 *
 * Time Complexity: O(M*M) in the worst case, due to the nested loops where `check` calls `isPrime`.
 * The outer loop in `check` runs up to `M-N` times, and `isPrime(M)` runs up to `M` times.
 * Space Complexity: O(1).
 */
#include <cstdio>

int n, m;

bool check(int n, int m);
bool isPrime(int n);

int main(void) {
  scanf("%d %d", &n, &m);
  // The core logic check is invoked and the result is printed.
  puts(check(n + 1, m) ? "YES" : "NO");
  return 0;
}

/**
 * @brief Checks if 'm' is the next prime after the range starting from 'n'.
 * @param n The start of the range (exclusive) to check for primes.
 * @param m The number to be verified as the next prime.
 * @return True if 'm' is prime and no primes exist in [n, m-1], otherwise false.
 */
bool check(int n, int m) {
  // First condition: 'm' must be a prime number.
  if (!isPrime(m))
    return false;

  /**
   * Second condition: no prime numbers should exist between the starting
   * number and 'm'.
   * Pre-condition: 'n' is the number after the input 'n'. 'm' is prime.
   * Invariant: The loop checks each number from 'n' to 'm-1' for primality.
   */
  while (n < m)
    if (isPrime(n++))
      return false; // A prime was found before 'm', so 'm' is not the *next* prime.

  return true;
}

/**
 * @brief A simple trial-division primality test.
 * @param n The integer to test.
 * @return True if 'n' is prime, otherwise false.
 */
bool isPrime(int n) {
  // A number less than 2 is not prime. This check is implicitly handled by the loop condition.
  for (int k = 2; k < n; ++k)
    if (n % k == 0)
      return false;

  return true;
}
