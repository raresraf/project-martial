/**
 * @file 282A.CPP
 * @brief Solution to a competitive programming problem: "A. Bit++".
 *
 * This program simulates the execution of a simple programming language called Bit++.
 * The language operates on a single variable 'x', initially 0, and supports two operations:
 * '++' (increment 'x' by 1) and '--' (decrement 'x' by 1). Statements are
 * 3-character strings like "++X", "X++", "--X", or "X--". The program reads
 * a sequence of these statements and outputs the final value of 'x' after
 * all statements are executed.
 *
 * Algorithm: Iterative statement processing with conditional value update.
 * Time Complexity: O(N), where N is the number of statements. Each statement
 *                  is processed in constant time.
 * Space Complexity: O(1) for storing variables, plus O(S_max) for the maximum
 *                   length of a statement string (which is constant, 3).
 */
#include <algorithm> // For various algorithms like std::min, std::max.
#include <cctype>    // For character manipulation functions like isalpha, isdigit.
#include <climits>   // For integer limits.
#include <cmath>     // For mathematical functions.
#include <cstdio>    // For C-style input/output functions like scanf.
#include <cstdlib>   // For general utilities.
#include <cstring>   // For C-style string manipulation functions.
#include <iostream>  // For C++ style input/output operations like cin, cout.
#include <list>      // For std::list container.
#include <map>       // For std::map container.
#include <queue>     // For std::queue container.
#include <set>       // For std::set container.
#include <stack>     // For std::stack container.
#include <string>    // For std::string class.
#include <vector>    // For std::vector container.

/**
 * @brief Standard namespace usage.
 * In competitive programming, 'using namespace std;' is common practice to avoid
 * repeatedly prefixing standard library elements with 'std::'.
 */
using namespace std;

// Type definitions for common data types, enhancing readability and brevity.
typedef unsigned long long ULL; ///< @brief Alias for unsigned long long integer.
typedef long long LL;           ///< @brief Alias for long long integer.

// Macros for common looping patterns, input/output, and utilities, optimizing competitive programming code.
#define REP(i, n) FOR(i, 0, n) ///< @brief Loop from 0 (inclusive) to 'n' (exclusive).
#define FOR(i, a, b) for (int i = a; i < b; i++) ///< @brief Loop from 'a' (inclusive) to 'b' (exclusive).
#define ROF(i, a, b) for (int i = a; i > b; i--) ///< @brief Loop from 'a' (inclusive) down to 'b'+1 (exclusive).
// Macro for fast integer input using scanf.
#define GI                                                                     \
  ({                                                                           \
    int t;                                                                     \
    scanf("%d", &t);                                                           \
    t;                                                                         \
  })
// Macro for fast long long input using cin.
#define GL                                                                     \
  ({                                                                           \
    LL t;                                                                      \
    cin >> t;                                                                  \
    t;                                                                         \
  })
// Macro for fast double input using scanf.
#define GD                                                                     \
  ({                                                                           \
    double t;                                                                  \
    scanf("%lf", &t);                                                          \
    t;                                                                         \
  })
#define pb push_back ///< @brief Appends an element to a std::vector.
#define mp make_pair ///< @brief Creates a std::pair.
#define MOD 1000000007 ///< @brief Constant for modulo operation.
#define INF (int)1e8 ///< @brief Integer infinity constant.
#define EPS 1e-9     ///< @brief Epsilon for floating-point comparisons.
// Macro for iterating through containers (type-safe for C++03, older style).
#define TR(a, it)                                                              \
  for (typeof((a).begin()) it = (a).begin(); it != (a).end(); ++it)

/**
 * @brief Main function: entry point of the program.
 * Reads the number of statements for the Bit++ program, then processes each statement
 * to update a counter representing the variable 'x', and finally prints the result.
 */
int main() {
  long long n = GL; // Reads the number of statements (N) using the GL macro.
  int count = 0;    // Initializes 'count', representing the variable 'x', to 0.

  /**
   * @brief Processes each statement in the Bit++ program.
   * Block Logic: Iterates 'n' times, reading one statement in each iteration.
   * It determines if the statement is an increment or decrement operation
   * by checking the first or third character for a '+'.
   * Pre-condition: 'n' holds the number of statements to process.
   * Invariant: 'count' accurately reflects the value of 'x' after processing
   *            all statements up to the current iteration.
   */
  FOR(i, 0, n) { // Loop 'n' times, once for each statement.
    string s;   // Declares a string to store the current statement.
    cin >> s;   // Reads the current statement (e.g., "++X", "X++", "--X", "X--").

    // Checks if the statement is an increment operation.
    // Statements are always 3 characters long. If either the first or third
    // character is '+', it indicates an increment.
    if (s[0] == '+' || s[2] == '+')
      count++; // Increment 'count' if the statement is '++X' or 'X++'.
    else
      count--; // Otherwise, it must be a decrement operation ('--X' or 'X--'), so decrement 'count'.
  };
  cout << count; // Prints the final value of 'x'.
  return 0;      // Indicates successful program execution.
}