/**
 * @file 69A.CPP
 * @brief Solution to a prime number checking problem.
 *
 * This program determines if a given integer `m` is the smallest prime number
 * strictly greater than another given integer `n`.
 *
 * Algorithm: Iterative prime checking and range search.
 * Time Complexity: O(M * sqrt(M)) in the worst case, due to repeated `isPrime` calls
 *                  where `isPrime` itself is O(N) (or O(sqrt(N)) if optimized, but here it's O(N)).
 *                  Specifically, `check` iterates `M-N` times, each calling `isPrime` up to `M` times.
 * Space Complexity: O(1).
 */
#include <cstdio>

int n, m;

bool check(int n, int m);
bool isPrime(int n);

/**
 * @brief Main function to read inputs and determine if 'm' is the next prime after 'n'.
 *
 * Functional Utility: Reads two integers `n` and `m`, then calls `check` function
 * with `n+1` (to search for prime strictly greater than `n`) and `m`.
 * Prints "YES" or "NO" based on the result.
 */
int main(void) {
  scanf("%d %d", &n, &m); // Read integers n and m.
  // Call 'check' to determine if 'm' is the smallest prime strictly greater than 'n'.
  // The 'check' function will implicitly check primality of m, and then search for
  // any other primes between (n+1) and m.
  puts(check(n + 1, m) ? "YES" : "NO");
  return 0;
}

/**
 * @brief Checks if 'm' is the smallest prime number strictly greater than 'n'.
 *
 * This function first verifies if 'm' is a prime number. If not, it returns false.
 * Then, it iterates through numbers from `n+1` up to `m-1` to ensure no other
 * prime numbers exist in this range.
 *
 * @param n The lower bound (exclusive) for checking intermediate primes.
 * @param m The candidate prime number and upper bound (exclusive) for intermediate prime check.
 * @return `true` if `m` is prime and there are no primes between `n` and `m`, `false` otherwise.
 */
bool check(int n, int m) {
  // Functional Utility: First check if 'm' itself is a prime number.
  // If 'm' is not prime, it cannot be the next prime number.
  if (!isPrime(m))
    return false;

  /**
   * Block Logic: Iterates through numbers from 'n' up to 'm-1'.
   * It checks if any of these intermediate numbers are prime.
   * If a prime is found, it means 'm' is not the *smallest* prime greater than the original 'n',
   * so the function returns false.
   * Pre-condition: 'm' has been confirmed as a prime number.
   * Invariant: For each iteration, `n` represents a number being tested for primality.
   */
  while (n < m)
    if (isPrime(n++)) // Check if 'n' (and then 'n+1', etc.) is prime.
      return false;   // If an intermediate prime is found, 'm' is not the next prime.

  // If 'm' is prime and no intermediate primes were found, then 'm' is the next prime.
  return true;
}

/**
 * @brief Checks if a given integer is a prime number.
 *
 * This function implements a basic primality test using trial division.
 * It checks for divisibility by all integers from 2 up to `n-1`.
 * Note: This implementation is inefficient for large numbers. For optimization,
 * it could check divisibility only up to `sqrt(n)`.
 *
 * @param n The integer to test for primality.
 * @return `true` if `n` is prime, `false` otherwise.
 */
bool isPrime(int n) {
  /**
   * Block Logic: Iterates from `k=2` up to `n-1`.
   * If `n` is divisible by any `k` in this range, `n` is not prime.
   * Pre-condition: `n` is a positive integer.
   * Invariant: `k` is a potential divisor being tested.
   */
  for (int k = 2; k < n; ++k)
    if (n % k == 0) // If 'n' is divisible by 'k', it's not prime.
      return false;

  // If no divisors were found, 'n' is prime.
  return true;
}
