/**
 * @file 898B.CPP
 * @brief Solution for Codeforces Problem 898B - Proper Nutrition.
 *
 * This problem asks to determine if a given integer 'n' can be represented as a sum
 * of multiples of two other integers 'p' and 'q' (i.e., n = x*p + y*q, where x and y are non-negative integers).
 * If such non-negative integers x and y exist, the solution should output "YES" and one such pair (x, y).
 * Otherwise, it should output "NO".
 *
 * Algorithm: Brute-force search (iterative approach).
 * The algorithm iterates through possible non-negative integer values for 'x' (represented by 'i')
 * starting from 0, up to a maximum where `i * p` does not exceed `n`. For each 'i', it calculates
 * the remaining value `k = n - (p * i)`. If `k` is non-negative and perfectly divisible by `q`,
 * then a valid non-negative integer 'y' (represented by `k / q`) is found.
 *
 * Time Complexity: O(N/P), where N is the target sum and P is the smaller of the two multipliers.
 * In the worst case, this is O(N) if P is 1.
 * Space Complexity: O(1) for the main logic, excluding potential auxiliary space for template functions.
 * Problem Tags: brute force, number theory.
 */

/*
        SubmissionId	:	33600243
        ContestId	:	898
        Index	:	B
        ProblemName	:	Proper Nutrition
        ProblemTags	:	['brute force', 'number theory']
        ProgrammingLanguage	:	GNU C++14
        Verdict	:	OK
*/

#include <bits/stdc++.h> // Includes most standard C++ libraries, common in competitive programming.

typedef long long ll; ///< Type definition for long long, commonly used for larger integers.
typedef unsigned long long ull; ///< Type definition for unsigned long long, used for larger unsigned integers.

/**
 * @brief Macro for the value of PI.
 * Functional Utility: Provides a high-precision constant for mathematical calculations involving circles or angles.
 */
#define pi                                                                     \
  3.141592653589793238462643383279502884197169399375105820974944592307816406286
/**
 * @brief Modulo constant.
 * Functional Utility: Often used in number theory problems for operations that need to wrap around a certain value,
 * such as in exponentiation by squaring to prevent integer overflow.
 */
#define mod 1000000007

using namespace std;

/**
 * @brief Checks if a given unsigned long long number is prime.
 * Algorithm: Optimized trial division. It checks divisibility by 2 and 3, then
 * iterates through numbers of the form 6k ± 1 up to the square root of n.
 * Time Complexity: O(sqrt(n))
 * @param n The number to check for primality.
 * @return `true` if `n` is prime, `false` otherwise.
 */
bool isprime(ull n) {
  if (n <= 1)
    return false; // Numbers less than or equal to 1 are not prime.
  if (n <= 3)
    return true; // 2 and 3 are prime.
  if (n % 2 == 0 || n % 3 == 0)
    return false; // Multiples of 2 or 3 are not prime (except 2,3 themselves).
  /**
   * Block Logic: Iterate with an optimized step to check for divisibility.
   * Invariant: `i` takes values 5, 11, 17, ... and `i+2` takes values 7, 13, 19, ...
   *            (i.e., numbers of the form 6k ± 1).
   */
  for (ull i = 5; i * i <= n; i = i + 6)
    if (n % i == 0 || n % (i + 2) == 0)
      return false; // If divisible, it's not prime.
  return true; // If no divisors found, it's prime.
}

/**
 * @brief Computes the greatest common divisor (GCD) of two numbers using the Euclidean algorithm.
 * Algorithm: Euclidean algorithm (recursive).
 * Time Complexity: O(log(min(a, b)))
 * @param a The first number.
 * @param b The second number.
 * @return The greatest common divisor of a and b.
 */
long long int gcd(long long int a, long long int b) {
  if (b == 0)
    return a; // Base case: GCD(a, 0) = a.
  else
    return gcd(b, a % b); // Recursive step: GCD(a, b) = GCD(b, a % b).
}

/**
 * @brief Computes (value ^ power) % mod using exponentiation by squaring.
 * Algorithm: Binary exponentiation (exponentiation by squaring).
 * Time Complexity: O(log(power))
 * @param value The base.
 * @param power The exponent.
 * @return (value ^ power) % mod.
 */
long long int exp(long long int value, long long int power) {
  long long int result = 1; // Initialize result to 1.
  /**
   * Block Logic: Iterate while power is greater than 0.
   * Invariant: `result` accumulates `value` raised to the bits already processed in `power`.
   */
  while (power > 0) {
    if (power & 1LL) { // If the current bit of power is 1.
      result *= value; // Multiply result by value.
      result %= mod; // Apply modulo.
    }
    value *= value; // Square the value.
    value %= mod; // Apply modulo.
    power /= 2LL; // Divide power by 2 (right shift).
  }
  return result; // Return the final result.
}

/**
 * @brief Computes the least common multiple (LCM) of two numbers.
 * Algorithm: Iterative search starting from the maximum of the two numbers.
 * Time Complexity: Inefficient for large numbers, could be O(a*b/gcd(a,b)) if implemented using GCD.
 * @param n1 The first number.
 * @param n2 The second number.
 * @return The least common multiple of n1 and n2.
 */
long long int lcm(ll n1, ll n2) {
  ll max =
      (n1 > n2) ? n1 : n2; // maximum value between n1 and n2 is stored in max
  /**
   * Block Logic: Iterate indefinitely until LCM is found.
   * Invariant: `max` is continuously incremented until it's a multiple of both `n1` and `n2`.
   */
  do {
    if (max % n1 == 0 && max % n2 == 0) {
      return max; // Return max when it's a common multiple.
    } else
      ++max; // Increment max.
  } while (true);
}

/**
 * @brief Checks if a given long long number is prime (a variation of the earlier function).
 * Algorithm: Optimized trial division. It checks divisibility by 2 and 3, then
 * iterates through numbers of the form 6k ± 1 up to the square root of n.
 * Time Complexity: O(sqrt(n))
 * @param n The number to check for primality.
 * @return 1 if `n` is prime, 0 otherwise.
 */
int isprime(ll n) {
  if (n == 1)
    return 0; // 1 is not prime.
  if (n <= 3)
    return 1; // 2 and 3 are prime.
  if (n % 2 == 0 || n % 3 == 0)
    return 0; // Multiples of 2 or 3 are not prime.
  /**
   * Block Logic: Iterate with an optimized step to check for divisibility.
   * Invariant: `i` takes values 5, 11, 17, ... and `i+2` takes values 7, 13, 19, ...
   *            (i.e., numbers of the form 6k ± 1).
   */
  for (ll i = 5; i * i <= n; i += 6) {
    if (n % i == 0 || n % (i + 2) == 0)
      return 0; // If divisible, it's not prime.
  }
  return 1; // If no divisors found, it's prime.
}

ll a[1000005]; ///< Global array `a`, likely used as a temporary buffer or for storing digits.
ll b[1000005]; ///< Global array `b`, not used in this specific solution.
ll c[1000005]; ///< Global array `c`, not used in this specific solution.
ll d[1000005]; ///< Global array `d`, not used in this specific solution.

/**
 * @brief Checks if a given long long number is a palindrome.
 * Algorithm: Extracts digits of the number and stores them in an array, then
 * compares digits from the beginning and end of the array.
 * Time Complexity: O(log10(n))
 * @param n The number to check for palindromic property.
 * @return `true` if `n` is a palindrome, `false` otherwise.
 */
bool ispalin(ll n) {
  ll k = 0; // Index for the array `a`.
  /**
   * Block Logic: Extract digits from `n` and store them in reverse order in array `a`.
   * Invariant: `n` is progressively reduced by dividing by 10, and its digits are stored.
   */
  while (n > 0) {
    a[k] = n % 10; // Get the last digit.
    k += 1; // Increment index.
    n /= 10; // Remove the last digit.
  }
  /**
   * Block Logic: Compare digits from both ends of the array to check for palindrome.
   * Invariant: `a[i]` is compared with `a[k - i - 1]` for the first half of the digits.
   */
  for (ll i = 0; i < (k / 2); i++) {
    if (a[i] != a[k - i - 1])
      return false; // If digits don't match, it's not a palindrome.
  }
  return true; // If all digits match, it's a palindrome.
}

/**
 * @brief Main function: Solves the Proper Nutrition problem.
 * Functional Utility: Reads three integers n, p, q. Determines if n can be
 * expressed as x*p + y*q for non-negative integers x, y. If so, prints "YES"
 * and one such pair (x, y); otherwise, prints "NO".
 * Algorithm: Brute-force iteration over possible values of x.
 * Time Complexity: O(n/p)
 */
int main() {
  /**
   * Functional Utility: Optimizes C++ standard streams for faster input/output operations.
   * `ios_base::sync_with_stdio(0)` unties C++ streams from C standard streams.
   * `cin.tie(NULL)` unties `cin` from `cout`, preventing flushes before input operations.
   * `cout.tie(NULL)` unties `cout` from `cin`, preventing flushes before output operations.
   */
  ios_base::sync_with_stdio(0);
  cin.tie(NULL);
  cout.tie(NULL);
  long long int min_, max_, ans, n, sum, f, j, cnt, len, row, col, p, q, r, t,
      i, k, l, m, x, y, z; // Declare variables, some are likely leftover from a template.
  cin >> n >> p >> q; // Read the input values: target sum n, and multipliers p, q.
  // The problem is to find non-negative integers i (for x) and j (for y) such that p*i + q*j = n.
  f = 0; // Flag to indicate if a solution is found (0 = no, 1 = yes).
  /**
   * Block Logic: Iterate through possible non-negative values for `i` (representing x).
   * Precondition: `n`, `p`, `q` are positive integers.
   * Invariant: `i * p` never exceeds `n`.
   * Postcondition: If a solution is found, `f` is set to 1 and `i` and `k/q` are printed.
   */
  for (i = 0; i <= (n / p); i++) {
    k = n - (p * i); // Calculate the remaining amount after accounting for `p * i`.
    /**
     * Block Logic: Check if the remaining amount `k` can be perfectly formed by multiples of `q`.
     * Precondition: `k` must be non-negative.
     */
    if (k % q == 0 && k >= 0) { // If k is divisible by q and k is non-negative.
      f = 1; // Set flag to indicate solution found.
      cout << "YES\n"; // Print "YES".
      cout << i << " " << k / q; // Print the pair (x, y) as (i, k/q).
      break; // Exit the loop as one solution is sufficient.
    }
  }
  /**
   * Block Logic: If no solution was found after iterating through all possibilities.
   * Precondition: The loop for `i` has completed without finding a solution (`f` remains 0).
   */
  if (f == 0)
    cout << "NO"; // Print "NO".

  cout << endl; // Ensure a newline at the end of output.
  return 0; // Indicate successful program execution.
}

/*
freopen("input.txt","r",stdin);
freopen("output.txt","w",stdout);
*/
