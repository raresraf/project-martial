/**
 * @file 118A.CPP
 * @brief Solution for Codeforces Problem 118A - String Task.
 *
 * Algorithm: Iterates through an input string, transforming it based on specific rules.
 * For each character, it checks if it's a vowel. If not, it prepends a '.' and appends
 * the lowercase version of the character to the output.
 *
 * Time Complexity: O(N) where N is the length of the input string, as it involves a single pass.
 * Space Complexity: O(N) to store the input string and potentially the output string.
 */

// Codeforces 118A - String Task [Optimized].cpp
#include <cstdio> // For standard input/output operations (scanf, printf).
#include <iostream> // For standard C++ input/output operations (cin, cout).

#include <cstring> // For C-style string manipulation functions.
#include <sstream> // For string stream manipulation (e.g., converting between strings and numbers).
#include <string> // For C++ string class.

#include <deque> // For deque container.
#include <list> // For list container.
#include <map> // For map container.
#include <queue> // For queue container.
#include <set> // For set container.
#include <stack> // For stack container.
#include <vector> // For vector container.

#include <algorithm> // For various algorithms (e.g., sort, min, max).
#include <bitset> // For bitset operations.
#include <functional> // For function objects.
#include <numeric> // For numeric operations (e.g., accumulate).
#include <utility> // For utility pairs.

#include <cctype> // For character classification functions (e.g., isalpha, isdigit).
#include <cmath> // For mathematical functions (e.g., sqrt, pow).
#include <cstdlib> // For general utilities (e.g., atoi, abs).
#include <ctime> // For time-related functions.
#include <iomanip> // For output formatting.

/**
 * @brief Global variable to track case numbers for competitive programming output.
 */
int caseno = 1;
#define NL '\n' ///< Macro for newline character.
#define FOR(I, J, K) for (I = J; I < K; I++) ///< Macro for a forward loop.
#define REV(I, J, K) for (I = J; I > K; I--) ///< Macro for a reverse loop.
#define sf1(x) scanf("%d", &x) ///< Macro for scanning a single integer.
#define sf scanf ///< Macro for scanf.
#define pf1(x) printf("%d", x) ///< Macro for printing a single integer.
#define pf printf ///< Macro for printf.
#define PC() printf("Case %d: ", caseno++) ///< Macro for printing case number and incrementing.
#define READ() freopen("in.txt", "r", stdin) ///< Macro for redirecting standard input from "in.txt".
#define WRITE() freopen("out.txt", "w", stdout) ///< Macro for redirecting standard output to "out.txt".
using namespace std;

// BEGINTEMPLATE_BY_ZAYED
typedef unsigned long long ULL;    ///< Type definition for unsigned long long, used for large unsigned integers.
typedef unsigned long long uint64; ///< Type definition for unsigned long long, used for 64-bit unsigned integers.
typedef long long LL;              ///< Type definition for long long, used for large signed integers.
typedef long long int64;           ///< Type definition for long long, used for 64-bit signed integers.

/**
 * @brief Checks if 'b' is greater than 'a' and updates 'a' if true.
 * @tparam T The type of the variables, must support comparison.
 * @param a Reference to the variable to be updated.
 * @param b Value to compare with.
 * Functional Utility: Efficiently finds the maximum between two values and updates the first.
 */
template <class T> inline void checkmax(T &a, T b) {
  if (b > a)
    a = b;
} // NOTES:checkmax(
/**
 * @brief Checks if 'b' is less than 'a' and updates 'a' if true.
 * @tparam T The type of the variables, must support comparison.
 * @param a Reference to the variable to be updated.
 * @param b Value to compare with.
 * Functional Utility: Efficiently finds the minimum between two values and updates the first.
 */
template <class T> inline void checkmin(T &a, T b) {
  if (b < a)
    a = b;
} // NOTES:checkmin(
/**
 * @brief Returns the maximum of two values.
 * @tparam T The type of the values, must support comparison.
 * @param a The first value.
 * @param b The second value.
 * @return The greater of the two values.
 * Functional Utility: A generic function to determine the maximum value.
 */
template <class T> inline T MAX(T a, T b) { return (a > b) ? a : b; }
/**
 * @brief Returns the minimum of two values.
 * @tparam T The type of the values, must support comparison.
 * @param a The first value.
 * @param b The second value.
 * @return The smaller of the two values.
 * Functional Utility: A generic function to determine the minimum value.
 */
template <class T> inline T MIN(T a, T b) { return (a < b) ? a : b; }
/**
 * @brief Returns the square of a value.
 * @tparam T The type of the value, must support multiplication.
 * @param x The value to square.
 * @return The square of x.
 * Functional Utility: A generic function to calculate the square of a number.
 */
template <class T> inline T sqr(T x) { return x * x; } // NOTES:square
/**
 * @brief Returns the absolute value of a number.
 * @tparam T The type of the value, must support negation and comparison with 0.
 * @param a The value.
 * @return The absolute value of a.
 * Functional Utility: A generic function to calculate the absolute value.
 */
template <class T> inline T fABS(T a) { return a < 0 ? a * (-1) : a; }
/**
 * @brief Swaps the values of two variables.
 * @tparam T The type of the variables.
 * @param a Reference to the first variable.
 * @param b Reference to the second variable.
 * Functional Utility: A generic function to exchange the values of two variables.
 */
template <class T> inline void SWAP(T &a, T &b) {
  T t = a;
  a = b;
  b = t;
}
/**
 * @brief Calculates base raised to the power of exponent.
 * @param base The base number.
 * @param power The exponent.
 * @return The result of base^power.
 * Functional Utility: Computes integer powers using iterative multiplication.
 * Time Complexity: O(power).
 */
inline LL POW(LL base, LL power) {
  LL I, res = base;
  if (power == 0)
    return 1;
  for (I = 0; I < power - 1; I++)
    res *= base;
  return res;
}
// Translator
/**
 * @brief Checks if a character is an uppercase letter.
 * @param c The character to check.
 * @return `true` if `c` is an uppercase letter (A-Z), `false` otherwise.
 * Functional Utility: Character classification based on ASCII range.
 */
bool isUpperCase(char c) { return c >= 'A' && c <= 'Z'; } // NOTES: isUpperCase(
/**
 * @brief Checks if a character is a lowercase letter.
 * @param c The character to check.
 * @return `true` if `c` is a lowercase letter (a-z), `false` otherwise.
 * Functional Utility: Character classification based on ASCII range.
 */
bool isLowerCase(char c) { return c >= 'a' && c <= 'z'; } // NOTES: isLowerCase(
/**
 * @brief Checks if a character is a vowel (case-insensitive, including 'y').
 * @param c The character to check.
 * @return `true` if `c` is a vowel, `false` otherwise.
 * Functional Utility: Custom vowel classification for specific problem constraints.
 */
bool isVowel(char c) {
  return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' ||
          c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' ||
          c == 'y' || c == 'Y');
} // NOTES: isVowel(
/**
 * @brief Checks if a character is an alphabet letter.
 * @param c The character to check.
 * @return `true` if `c` is a letter (A-Z or a-z), `false` otherwise.
 * Functional Utility: Character classification based on ASCII range for letters.
 */
bool isLetter(char c) {
  return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z';
} // NOTES: isLetter(
/**
 * @brief Checks if a character is a digit.
 * @param c The character to check.
 * @return `true` if `c` is a digit (0-9), `false` otherwise.
 * Functional Utility: Character classification based on ASCII range for digits.
 */
bool isDigit(char c) { return c >= '0' && c <= '9'; } // NOTES:isDigit(
/**
 * @brief Converts an uppercase character to lowercase.
 * @param c The character to convert.
 * @return The lowercase version of `c` if it's uppercase, otherwise `c` itself.
 * Functional Utility: Performs case conversion for alphabet characters.
 */
char toLowerCase(char c) {
  return (isUpperCase(c)) ? (c + 32) : c;
} // NOTES: toLowerCase(
/**
 * @brief Converts a lowercase character to uppercase.
 * @param c The character to convert.
 * @return The uppercase version of `c` if it's lowercase, otherwise `c` itself.
 * Functional Utility: Performs case conversion for alphabet characters.
 */
char toUpperCase(char c) {
  return (isLowerCase(c)) ? (c - 32) : c;
} // NOTES: toUpperCase(
/**
 * @brief Converts a string to an integer.
 * @param s The string to convert.
 * @return The integer representation of the string.
 * Functional Utility: Parses a string to its integer equivalent using stringstreams.
 */
int toInt(string s) {
  int r = 0;
  istringstream sin(s);
  sin >> r;
  return r;
} // NOTES:toInt(
/**
 * @brief Converts a string to a double.
 * @param s The string to convert.
 * @return The double representation of the string.
 * Functional Utility: Parses a string to its double-precision floating-point equivalent using stringstreams.
 */
double toDouble(string s) {
  double r = 0;
  istringstream sin(s);
  sin >> r;
  return r;
} // NOTES:toDouble(
/**
 * @brief Converts an integer to a string.
 * @param n The integer to convert.
 * @return The string representation of the integer.
 * Functional Utility: Converts an integer to its string equivalent using stringstreams.
 */
string toString(int n) {
  string s;
  stringstream convert;
  convert << n;
  s = convert.str();
  return s;
} // NOTES:toString
// ENDTEMPLATE_BY_ZAYED

#define INF 2147483647 ///< Macro for a large integer value, typically representing infinity.
#define MOD 1000000007 ///< Macro for a modulus value, often used in modular arithmetic.
const double PI = 2 * acos(0.0); ///< Constant for the mathematical value of PI.
const double EPS = 1e-11; ///< Constant for a small epsilon value, used for floating-point comparisons.
const int SIZE = 1e6; ///< Constant for a maximum size, possibly for arrays or data structures.

// fflush(stdout);
string S; ///< Global string variable to store the input.

/**
 * @brief Main function: Processes the input string according to the problem rules.
 * Functional Utility: Reads a string, converts all non-vowel characters to lowercase,
 * prepends a '.' to them, and prints the modified string.
 * Algorithm: Iterates through the input string, applying conditional character transformation.
 */
int main() {
  /**
   * Functional Utility: Decouples C++ streams from standard C streams and unties cin from cout.
   * This optimizes high-throughput I/O, which is crucial for competitive programming constraints.
   */
  std::ios_base::sync_with_stdio(0);
  int tcases, I, J, K, N, n, m, cnt = 0, len; // Declare variables for potential use (some may be from template).
  // READ(); ///< Macro to redirect standard input from "in.txt" (commented out).
  // WRITE(); ///< Macro to redirect standard output to "out.txt" (commented out).
  cin >> S; // Read the input string into S.
  len = S.length(); // Get the length of the input string.

  /**
   * Block Logic: Iterate through each character of the input string S.
   * Precondition: S is a valid string.
   * Invariant: Characters up to index I-1 have been processed.
   * Logic: For each character, check if it's a vowel using `isVowel()`.
   */
  for (I = 0; I < len; I++) {
    /**
     * Block Logic: If the current character is NOT a vowel, process it.
     * Precondition: Current character `S[I]` is accessible.
     */
    if (!isVowel(S[I])) {
      cout << "."; // Print a dot.
      /**
       * Block Logic: Convert the character to lowercase and print it.
       * Logic: If the character is already lowercase, print it directly. Otherwise, convert and print.
       */
      if (isLowerCase(S[I]))
        cout << S[I]; // Character is already lowercase.
      else
        cout << toLowerCase(S[I]); // Convert to lowercase and print.
    }
  }
  cout << NL; // Print a newline character at the end of the output.
  return 0; // Indicate successful execution.
}