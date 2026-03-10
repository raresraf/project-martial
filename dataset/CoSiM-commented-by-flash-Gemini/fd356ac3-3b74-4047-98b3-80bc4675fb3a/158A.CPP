/**
 * @file 158A.CPP
 * @brief This program solves the "Next Round" problem (Codeforces 158A).
 *   It counts the number of participants who qualify for the next round of a competition.
 *   Qualification criteria: a participant must score at least as much as the `k`-th placed
 *   participant and must have a positive score.
 *   Problem description: https://codeforces.com/problemset/problem/158/A
 * * Algorithm: Array traversal and conditional counting.
 *   1. Reads `n` participant scores into an array.
 *   2. Handles a special edge case where all scores might be zero.
 *   3. Determines the cutoff score based on the `k`-th participant's score.
 *   4. Iterates through all scores, applying the qualification criteria (score >= cutoff AND score > 0)
 *      to count qualifiers.
 * Time Complexity: O(N) where N is the number of participants. Input reading,
 *   the all-zeros check, and the counting loop are all linear with respect to N.
 * Space Complexity: O(N) for storing the global `a` array.
 */
// Codeforces 158A - Next Round.cpp
#include <cstdio>     // Required for scanf, printf.
#include <iostream>   // Required for cin, cout (though not used in main).

#include <cstring>  // For string manipulation, often included.
#include <sstream>  // For string stream operations, often included.
#include <string>   // For std::string, often included.

#include <deque>    // Standard library container, often included.
#include <list>     // Standard library container, often included.
#include <map>      // Standard library container, often included.
#include <queue>    // Standard library container, often included.
#include <set>      // Standard library container, often included.
#include <stack>    // Standard library container, often included.
#include <vector>   // Standard library container.

#include <algorithm> // Required for std::sort, std::max, std::min etc.
#include <bitset>    // For bitset operations, often included.
#include <functional> // For function objects, often included.
#include <numeric>   // For numeric operations, often included.
#include <utility>   // For std::pair, often included.

#include <cctype> // For character classification, often included.
#include <cmath>  // For mathematical functions, often included.
#include <cstdlib> // For general utilities, often included.
#include <ctime>  // For time functions, often included.
#include <iomanip> // For I/O manipulators, often included.

// Functional Utility: Macro for `long long`.
#define LL long long
// Constant: Defines a maximum size, likely for array bounds.
#define MAX 100005
// Constant: For modulo operations, common in number theory problems.
#define MOD 1000000007

// Functional Utility: Macro for newline character.
#define NL '\n'
using namespace std;

// BEGIN TEMPLATE: This section includes common competitive programming helper utilities.
const double PI = 2 * acos(0.0); // Constant for Pi.
const double EPS = 1e-11;        // Constant for Epsilon, used for floating point comparisons.
const int sz = 1e5 + 10;         // Another size constant.

// Functional Utility: Typedef for `unsigned long long`.
typedef unsigned long long ull;
// Functional Utility: Typedef for `long long`.
typedef long long ll;
// Functional Utility: Typedef for `long long int` (same as `long long`).
typedef long long int64;
// Functional Utility: Typedef for `unsigned long long`.
typedef unsigned long long uint64;

// Functional Utility: Typedef for `std::stack<int>`.
typedef stack<int> si;
// Functional Utility: Typedef for `std::queue<int>`.
typedef queue<int> qi;
// Functional Utility: Typedef for `std::vector<int>`.
typedef vector<int> vi;
// Functional Utility: Typedef for `std::pair<ll, ll>`.
typedef pair<ll, ll> pll;
// Functional Utility: Typedef for `std::map<int, int>`.
typedef mii map<int, int>;
// Functional Utility: Typedef for `std::pair<int, int>`.
typedef pii pair<int, int>;
// Functional Utility: Typedef for `std::map<char, int>`.
typedef mci map<char, int>;
// Functional Utility: Typedef for `std::map<std::string, int>`.
typedef msi map<string, int>;
// Functional Utility: Typedef for `std::pair<std::string, int>`.
typedef psi pair<string, int>;
// Functional Utility: Typedef for `std::map<std::string, std::string>`.
typedef mss map<string, string>;

/**
 * @brief Functional Utility: Updates `a` with `b` if `b` is smaller than `a`.
 * @param a Reference to a variable to be updated.
 * @param b Value to compare with.
 */
template <class T> inline void checkmin(T &a, T b) {
  if (b < a)
    a = b;
}
/**
 * @brief Functional Utility: Updates `a` with `b` if `b` is larger than `a`.
 * @param a Reference to a variable to be updated.
 * @param b Value to compare with.
 */
template <class T> inline void checkmax(T &a, T b) {
  if (b > a)
    a = b;
}
/**
 * @brief Functional Utility: Calculates the square of a value.
 * @param x The value to square.
 * @return The square of `x`.
 */
template <class T> inline T sqr(T x) { return x * x; }
/**
 * @brief Functional Utility: Returns the maximum of two integers.
 * @param a First integer.
 * @param b Second integer.
 * @return The larger of `a` and `b`.
 */
int _max(int a, int b) { return (a > b) ? a : b; }
/**
 * @brief Functional Utility: Returns the minimum of two integers.
 * @param a First integer.
 * @param b Second integer.
 * @return The smaller of `a` and `b`.
 */
int _min(int a, int b) { return (a < b) ? a : b; }
/**
 * @brief Functional Utility: Calculates integer power.
 * @param base The base number.
 * @param power The exponent.
 * @return `base` raised to the power of `power`.
 */
inline int _pow(int base, int power) {
  int I, result = base;
  if (power == 0)
    return 1;
  for (I = 0; I < power - 1; I++) {
    result = result * base;
  }
  return result;
}
// Functional Utility: Checks if a character is uppercase.
bool isUpperCase(char c) { return c >= 'A' && c <= 'Z'; }
// Functional Utility: Checks if a character is lowercase.
bool isLowerCase(char c) { return c >= 'a' && c <= 'z'; }
// Functional Utility: Checks if a character is an alphabet letter.
bool isLetter(char c) {
  return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z';
}
// Functional Utility: Checks if a character is a digit.
bool isDigit(char c) { return c >= '0' && c <= '9'; }
// Functional Utility: Converts an uppercase character to lowercase.
char toLowerCase(char c) {
  return (isUpperCase(c)) ? (c + 32) : c;
}
// Functional Utility: Converts a lowercase character to uppercase.
char toUpperCase(char c) {
  return (isLowerCase(c)) ? (c - 32) : c;
}
// Functional Utility: Converts a string to an integer.
int toInt(string s) {
  int r = 0;
  istringstream sin(s);
  sin >> r;
  return r;
}
// Functional Utility: Converts a string to a double.
double toDouble(string s) {
  double r = 0;
  istringstream sin(s);
  sin >> r;
  return r;
}
// END TEMPLATE
// Global array to store scores. Size 110 implies N <= 109.
int a[110];

/**
 * @brief Main function to count qualifying participants.
 * Functional Utility: Reads participant count, cutoff rank, and scores, then
 *                     applies qualification rules to determine and print the number of qualifiers.
 */
int main() {
  // std::ios_base::sync_with_stdio(0); // Commented out I/O optimization.
  int cases, caseno = 0, n, I, K, k, cnt = 0;
  // `n`: total participants, `k`: 1-indexed rank for cutoff, `cnt`: number of qualifiers.
  // `I`: loop counter, `cases/caseno/K`: unused variables.
  // Block Logic: Reads `n` (number of participants) and `k` (cutoff rank).
  scanf("%d%d", &n, &k);
  /**
   * Block Logic: Reads `n` participant scores into the global array `a`.
   * Pre-condition: `n` is a positive integer.
   * Invariant: `a` is populated with all participant scores (1-indexed).
   */
  for (I = 1; I <= n; I++)
    scanf("%d", &a[I]);

  /**
   * Block Logic: Checks if all scores are zero. This is a specific edge case.
   * If all scores are zero, it implies that the `k`-th score `a[k]` would also be zero,
   * and the qualification criteria `a[I] != 0` would fail for everyone.
   * Invariant: `I` will stop at the first non-zero score, or `n+1` if all are zero.
   */
  for (I = 1; I <= n; I++) {
    if (a[I] != 0) // Break if a non-zero score is found.
      break;
  }
  // Conditional Logic: If `I - 1 == n`, it means the loop completed, and all scores were zero.
  if (I - 1 == n) {
    printf("%d\n", cnt); // Prints 0 as no one can qualify with zero scores.
    return 0;             // Exits the program.
  }

  /**
   * Block Logic: Iterates through all participant scores to count qualifiers.
   * Pre-condition: `a` contains scores, `k` defines the cutoff rank, `a[k]` is the cutoff score.
   * Invariant: `cnt` increments for each participant meeting the criteria.
   */
  for (I = 1; I <= n; I++) {
    /**
     * Conditional Logic: Checks two conditions for qualification:
     *   1. `a[I] >= a[k]`: The participant's score `a[I]` must be greater than or equal to
     *      the score of the `k`-th placed participant (`a[k]`).
     *   2. `a[I] != 0`: The participant's score must be non-zero (positive, as scores are non-negative).
     */
    if (a[I] >= a[k] && a[I] != 0)
      cnt++; // Increments the counter if both conditions are met.
  }
  // Block Logic: Prints the total count of qualifying participants.
  printf("%d\n", cnt);
  return 0;
}
