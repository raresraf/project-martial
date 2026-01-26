/**
 * @file 1140D.CPP
 * @brief Solution for a competitive programming problem involving summation of a series.
 *
 * The problem requires calculating the sum of products `i * (i + 1)` for `i` ranging
 * from 2 up to `n-1`, where `n` is an input integer.
 * This sum can be expressed as: `sum = 2*3 + 3*4 + ... + (n-1)*n`.
 *
 * Algorithm: Direct iterative summation of the series.
 * Time Complexity: O(N), where N is the input integer, as the loop runs `N-2` times.
 * Space Complexity: O(1), as only a constant number of variables are used.
 */

#include <bits/stdc++.h> // Includes common C++ standard library headers.
//#include <boost/multiprecision/cpp_int.hpp> // Commented out, but indicates potential use of large integer arithmetic.

/**
 * @brief Macro for `unsigned long long int` type.
 * Functional Utility: Provides a shorthand for a 64-bit unsigned integer type,
 *                     useful for handling large numbers in competitive programming.
 */
#define ll unsigned long long int
/**
 * @brief Macro for `long double` type.
 * Functional Utility: Provides a shorthand for a high-precision floating-point type.
 */
#define ld long double
/**
 * @brief Macro for `push_back` method of vectors.
 * Functional Utility: Simplifies adding elements to the end of a vector.
 */
#define pb push_back
/**
 * @brief Macro for the mathematical constant Pi.
 * Functional Utility: Provides a precise value for Pi.
 */
#define pi 3.1415926536
/**
 * @brief Macro for a common modulo value (10^9 + 7).
 * Functional Utility: Used in problems involving large numbers to prevent overflow
 *                     and calculate results within a specific range.
 */
#define mod 1e9 + 7
/**
 * @brief Macro for `make_pair` function.
 * Functional Utility: Simplifies creating `std::pair` objects.
 */
#define mp make_pair
/**
 * @brief Macro for `while (t--)` loop.
 * Functional Utility: A common idiom in competitive programming for iterating
 *                     through a given number of test cases.
 */
#define wt while (t--)
/**
 * @brief Macro for a `for` loop to fill an array.
 * Functional Utility: Provides a shorthand for iterating from 0 to n-1.
 * @note Requires `n` to be defined in scope.
 */
#define fillarray for (int i = 0; i < n; i++)
/**
 * @brief Macro for handling multiple test cases input.
 * Functional Utility: Simplifies the structure for competitive programming problems
 *                     with multiple test cases.
 */
#define testcases                                                              \
  ll t;                                                                        \
  cin >> t;                                                                    \
  while (t--)

using namespace std;
// using namespace boost::multiprecision; // Commented out, indicates potential use of boost library.

int main() {
  int n; // Declares an integer to store the input value.
  // Reads the integer `n` from standard input.
  cin >> n;

  ll sum = 0; // Initializes an unsigned long long integer to store the accumulated sum.

  /**
   * Block Logic: Iterates from `i = 2` to `n-1` to calculate the sum of `i * (i + 1)`.
   * Invariant: After each iteration, `sum` holds the accumulated sum up to the current `i`.
   */
  for (int i = 2; i < n; i++) {
    sum += i * (i + 1); // Adds the product `i * (i + 1)` to the sum.
  }

  // Prints the final calculated sum.
  cout << sum;
}
