/**
 * @file 71A.CPP
 * @brief Solution to Codeforces Problem 71A: "Way Too Long Words".
 *        The problem involves reading a list of words and, for each word longer than 10 characters,
 *        replacing it with a shorthand: the first letter, the count of characters between
 *        the first and last letter, and the last letter. Words 10 characters or shorter
 *        are printed as is.
 *
 * Algorithm: String manipulation and conditional formatting. For each word, its length is checked.
 *            Based on the length, either the original word is printed or a constructed shorthand.
 * Time Complexity: O(N * L) where N is the number of words and L is the maximum length of a word.
 *                  String operations (size, access) take O(1) on average for C++ strings.
 * Space Complexity: O(L_max) to store the longest word.
 */
#include "bits/stdc++.h"
/**
 * @brief Macro to optimize C++ standard stream I/O operations.
 * Functional Utility: Unties C++ standard streams from C standard streams (`ios::sync_with_stdio(0)`)
 * and unties `cin` from `cout` (`cin.tie(0)` and `cout.tie(0)`), leading to faster input/output,
 * which is crucial for competitive programming performance.
 */
#define IOS                                                                    \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
/**
 * @brief Macro for getting iterators to the beginning and end of a container.
 * Functional Utility: Simplifies passing container ranges to algorithms.
 */
#define all(v) v.begin(), v.end()
/**
 * @brief Macro for getting reverse iterators to the beginning and end of a container.
 * Functional Utility: Simplifies iterating a container in reverse.
 */
#define allre(v) v.rbegin(), v.rend()
/**
 * @brief Macro for setting precision for floating-point output.
 * Functional Utility: Formats floating-point numbers to a fixed number of decimal places.
 * @param x: The floating-point number.
 * @param y: The number of decimal places.
 */
#define sp(x, y) fixed << setprecision(y) << x
/**
 * @brief Macro for finding the maximum of three values.
 * Functional Utility: Provides a convenient way to get the maximum of three arguments.
 */
#define max3(a, b, c) max(a, max(b, c))
/**
 * @brief Macro for finding the minimum of three values.
 * Functional Utility: Provides a convenient way to get the minimum of three arguments.
 */
#define min3(a, b, c) min(a, min(b, c))
/**
 * @brief Macro for calculating the greatest common divisor (GCD) of two numbers.
 * Functional Utility: Uses the built-in `__gcd` function for GCD computation.
 */
#define GCD(a, b) __gcd(a, b)
/**
 * @brief Macro for calculating the least common multiple (LCM) of two numbers.
 * Functional Utility: Computes LCM using the formula (a * b) / GCD(a, b).
 */
#define LCM(a, b) ((a * b) / __gcd(a, b))
using namespace std;

/**
 * @brief Constant for the mathematical value of Pi.
 */
const double pi = acos(-1.0);
/**
 * @brief Constant for a small epsilon value, typically used for floating-point comparisons.
 */
const double EPS = 1e-6;
/**
 * @brief Constant for a modulo value, commonly used in problems requiring large number arithmetic.
 */
const int MOD = (int)1e9 + 7;

/**
 * @brief Custom comparison function for sorting in reverse order.
 * Functional Utility: Returns true if 'a' is greater than 'b', used for descending sorts.
 * @param a: First long long integer.
 * @param b: Second long long integer.
 * @return: True if a > b, False otherwise.
 */
bool Reverse(long long a, long long b) { return a > b; }

/**
 * @brief Main function: Reads input, processes words, and prints results.
 * Functional Utility: Orchestrates the word shortening logic as per the problem statement.
 * @return: 0 on successful execution.
 */
int main() {
  IOS int n;
  cin >> n;
  /**
   * Block Logic: Iterates 'n' times, processing one word in each iteration.
   * Pre-condition: 'n' is the total number of words to be processed.
   * Invariant: In each iteration, 's' contains the current word to be evaluated.
   */
  while (n--) {
    string s;
    cin >> s;
    /**
     * Conditional Logic: Checks if the word length exceeds 10 characters.
     * If the word is short (length <= 10), it's printed as is.
     * If the word is long (length > 10), it's transformed into a shorthand.
     */
    if (s.size() <= 10)
      cout << s << endl;
    else
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << endl;
  }
  return 0;
}
