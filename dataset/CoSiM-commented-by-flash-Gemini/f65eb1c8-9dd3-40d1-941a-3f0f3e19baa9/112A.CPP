/**
 * @file 112A.CPP
 * @brief This competitive programming solution performs a case-insensitive
 *        lexicographical comparison between two input strings.
 *        It outputs -1 if the first string is lexicographically smaller,
 *        0 if they are equal, and 1 if the first string is greater.
 * Algorithm: Custom Case Normalization and Standard String Comparison.
 *            1. Implements a `toLower` function that converts all uppercase letters
 *               in a string to lowercase by adjusting their ASCII values.
 *            2. Reads two input strings.
 *            3. Converts both strings to lowercase using the custom `toLower` function.
 *            4. Compares the resulting lowercase strings using standard C++ string
 *               comparison operators (`<`, `==`, `>`).
 * Time Complexity: O(L), where L is the maximum length of the two input strings.
 *                  This is because `toLower` iterates through each string once, and
 *                  string comparison also takes linear time.
 * Space Complexity: O(L) for storing the two input strings. The `toLower` function
 *                   operates on a copy of the string (due to pass-by-value for `string str`),
 *                   incurring O(L) additional space for the copy, which is then modified in-place.
 */
#include <algorithm> // Provides std::sort, etc. (some are unused here).
#include <cmath>     // Provides math functions (unused here).
#include <cstdlib>   // Provides general utilities (unused here).
#include <deque>     // Provides std::deque (unused here).
#include <fstream>   // Provides file stream operations (unused here).
#include <iostream>  // Required for standard input/output (cin, cout).
#include <limits>    // Provides numeric_limits (unused here).
#include <list>      // Provides std::list (unused here).
#include <map>       // Provides std::map (unused here).
#include <queue>     // Provides std::queue (unused here).
#include <set>       // Provides std::set (unused here).
#include <sstream>   // Provides std::stringstream (unused here).
#include <stack>     // Provides std::stack (unused here).
#include <string>    // Required for std::string.
#include <vector>    // Provides std::vector (unused here).

// using namespace std; // Using directive for brevity.

// typedefs for common types, useful in competitive programming.
typedef vector<int> VI;    ///< Alias for vector of integers.
typedef vector<VI> VII;   ///< Alias for vector of vectors of integers (2D array).
typedef vector<string> VS; ///< Alias for vector of strings.
typedef vector<long> VL;   ///< Alias for vector of longs.
typedef long long LL;     ///< Alias for long long integer.
typedef vector<LL> VLL;   ///< Alias for vector of long longs.

// container utils - macros for common container operations.
#define PB push_back   ///< Macro for `vector::push_back`.
#define PF push_front  ///< Macro for `deque::push_front`.
#define GRT(x) greater<(x)>() ///< Macro for `std::greater`.
#define ASORT(x) sort((x).begin(), (x).end()) ///< Macro for ascending sort.
#define DSORT(x, y) sort((x).begin(), (x).end(), greater<(y)>()) ///< Macro for descending sort.
#define FILL(x, y) fill((x).begin(), (x).end(), (y)) ///< Macro to fill container with value.
#define COPY(x, y)                                                             
  (y).clear();                                                                 
  copy((x).begin(), (x).end(), back_inserter(y)) ///< Macro to copy container contents.

// repetition - macros for common loop patterns.
#define FOR(i, a, b) for (int i = (a); i < (b); i++)     ///< Loop from a (inclusive) to b (exclusive).
#define FORL(i, a, b) for (long i = (a); i < (b); i++)   ///< Long version of FOR.
#define REP(i, n) FOR(i, 0, n)                          ///< Loop from 0 (inclusive) to n (exclusive).
#define REPL(i, n) FORL(i, 0, n)                        ///< Long version of REP.

// debug - macros for debugging output.
#define DUMP(a) cerr << #a << " = " << (a) << endl ///< Prints variable name and value to cerr.
#define DUMP2(a, b)                                                            
  cerr << #a << " = " << (a) << ", " << #b << " = " << (b) << endl ///< Prints two variables.
#define DUMP3(a, b, c)                                                         
  cerr << #a << " = " << (a) << ", " << #b << " = " << (b) << ", " << #c       
       << " = " << (c) << endl ///< Prints three variables.

/**
 * @brief Converts an input string to its lowercase equivalent.
 * @param str The input string to be converted. A copy is made due to pass-by-value.
 * @return The converted string with all uppercase letters changed to lowercase.
 * Functional Utility: Provides case normalization for string comparison.
 */
string toLower(string str) {
  // Block Logic: Iterates through each character of the string.
  // Invariant: After each iteration, characters up to `i` are converted to lowercase if they were uppercase.
  REP(i, str.size()) {
    // If the character is an uppercase letter.
    if (str[i] >= 'A' && str[i] <= 'Z') {
      // Converts uppercase to lowercase by adding the ASCII difference between 'a' and 'A'.
      // Inline: ('a' - 'A') is typically 32 in ASCII, shifting the character value.
      str[i] += ('a' - 'A');
    }
  }
  return str; // Returns the lowercase version of the string.
}

/**
 * @brief Reads two strings, converts them to lowercase, compares them lexicographically, and prints the result.
 * Functional Utility: Encapsulates the core logic for the case-insensitive string comparison problem.
 */
void solve() {
  string s, t; // Declares two strings for input.

  cin >> s >> t; // Reads the two strings from standard input.

  s = toLower(s); // Converts the first string to lowercase.
  t = toLower(t); // Converts the second string to lowercase.

  // Block Logic: Performs lexicographical comparison of the two lowercase strings.
  // Functional Utility: Determines the relative order of the strings.
  if (s < t)
    cout << -1 << endl; // If s is lexicographically smaller, print -1.
  else if (s == t)
    cout << 0 << endl; // If s is equal to t, print 0.
  else
    cout << 1 << endl; // If s is lexicographically larger, print 1.
}

/**
 * @brief Main function of the program. Calls the `solve` function.
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line argument strings.
 * Functional Utility: Serves as the entry point for execution.
 */
int main(int argc, char *argv[]) {
  solve(); // Calls the function that implements the problem's solution.

  return 0; // Indicates successful program termination.
}
