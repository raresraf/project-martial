/**
 * @file 231A.CPP
 * @brief Solution to Codeforces problem 231A - Team.
 *        This program determines the number of problems a programming team can implement.
 *        A problem is implemented if at least two out of three team members are sure about the solution.
 * Algorithm:
 *   1. Reads an integer 'n', representing the number of problems.
 *   2. For each problem, reads three integers (0 or 1) indicating each friend's certainty.
 *   3. If the sum of the three integers for a problem is 2 or more, it means at least two friends
 *      are sure, and thus the problem is counted as implemented.
 *   4. The total count of implemented problems is then printed.
 * Time Complexity: O(n), where n is the number of problems, as each problem requires constant time processing.
 * Space Complexity: O(1) for storing a few variables, plus O(N) for the M array which is declared with size 1000.
 */
// Codeforces 231A - Team.cpp
#include <cstdio>
#include <iostream>

#include <cstring>
#include <sstream>
#include <string>

#include <deque>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <vector>

#include <algorithm>
#include <bitset>
#include <functional>
#include <numeric>
#include <utility>

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>

#define LL long long          ///< Macro for long long data type for convenience.
#define MAX 100005            ///< Macro for a common maximum array size or limit.
#define MOD 1000000007        ///< Macro for a common modulo value, typically used in modular arithmetic.

#define NL '\n' ///< Macro for newline character.
using namespace std;

// BEGINTEMPLATE_BY_ZAYED
const double PI = 2 * acos(0.0); ///< Constant for Pi.
const double EPS = 1e-11;      ///< Constant for a small epsilon value, used in floating-point comparisons.
const int sz = 1e5 + 10;       ///< Constant for a common array size or limit.

typedef unsigned long long ull;    ///< Typedef for unsigned long long.
typedef long long ll;              ///< Typedef for long long.
typedef long long int64;           ///< Typedef for long long int64.
typedef unsigned long long uint64; ///< Typedef for unsigned long long uint64.

typedef stack<int> si; ///< Typedef for stack of integers.
typedef queue<int> qi; ///< Typedef for queue of integers.
typedef vector<int> vi; ///< Typedef for vector of integers.
typedef pair<ll, ll> pll; ///< Typedef for pair of long longs.
typedef map<int, int> mii; ///< Typedef for map with int keys and int values.
typedef pair<int, int> pii; ///< Typedef for pair of integers.
typedef map<char, int> mci; ///< Typedef for map with char keys and int values.
typedef map<string, int> msi; ///< Typedef for map with string keys and int values.
typedef pair<string, int> psi; ///< Typedef for pair of string and int.
typedef map<string, string> mss; ///< Typedef for map with string keys and string values.

template <class T> inline void checkmin(T &a, T b) {
  /**
   * @brief Updates 'a' with 'b' if 'b' is smaller than 'a'.
   * @tparam T Type of the variables.
   * @param a Reference to the variable to be updated.
   * @param b The value to compare against.
   */
  if (b < a)
    a = b;
}
template <class T> inline void checkmax(T &a, T b) {
  /**
   * @brief Updates 'a' with 'b' if 'b' is larger than 'a'.
   * @tparam T Type of the variables.
   * @param a Reference to the variable to be updated.
   * @param b The value to compare against.
   */
  if (b > a)
    a = b;
}
template <class T> inline T sqr(T x) {
  /**
   * @brief Calculates the square of a given value.
   * @tparam T Type of the value.
   * @param x The value to square.
   * @return T The square of x.
   */
  return x * x;
}
int _max(int a, int b) {
  /**
   * @brief Returns the maximum of two integers.
   * @param a The first integer.
   * @param b The second integer.
   * @return int The maximum of a and b.
   */
  return (a > b) ? a : b;
}
int _min(int a, int b) {
  /**
   * @brief Returns the minimum of two integers.
   * @param a The first integer.
   * @param b The second integer.
   * @return int The minimum of a and b.
   */
  return (a < b) ? a : b;
}
inline int _pow(int base, int power) {
  /**
   * @brief Calculates integer power (base^power).
   * @param base The base.
   * @param power The exponent.
   * @return int The result of base raised to the power.
   */
  int I, result = base;
  if (power == 0)
    return 1;
  for (I = 0; I < power - 1; I++) {
    result = result * base;
  }
  return result;
}
// Translator
bool isUpperCase(char c) {
  /**
   * @brief Checks if a character is an uppercase letter.
   * @param c The character to check.
   * @return bool True if c is uppercase, False otherwise.
   */
  return c >= 'A' && c <= 'Z';
}
bool isLowerCase(char c) {
  /**
   * @brief Checks if a character is a lowercase letter.
   * @param c The character to check.
   * @return bool True if c is lowercase, False otherwise.
   */
  return c >= 'a' && c <= 'z';
}
bool isLetter(char c) {
  /**
   * @brief Checks if a character is an alphabetic letter.
   * @param c The character to check.
   * @return bool True if c is a letter, False otherwise.
   */
  return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z';
}
bool isDigit(char c) {
  /**
   * @brief Checks if a character is a digit.
   * @param c The character to check.
   * @return bool True if c is a digit, False otherwise.
   */
  return c >= '0' && c <= '9';
}
char toLowerCase(char c) {
  /**
   * @brief Converts an uppercase character to lowercase.
   * @param c The character to convert.
   * @return char The lowercase version of c, or c itself if not uppercase.
   */
  return (isUpperCase(c)) ? (c + 32) : c;
}
char toUpperCase(char c) {
  /**
   * @brief Converts a lowercase character to uppercase.
   * @param c The character to convert.
   * @return char The uppercase version of c, or c itself if not lowercase.
   */
  return (isLowerCase(c)) ? (c - 32) : c;
}
int toInt(string s) {
  /**
   * @brief Converts a string to an integer.
   * @param s The string to convert.
   * @return int The integer representation of the string.
   */
  int r = 0;
  istringstream sin(s);
  sin >> r;
  return r;
}
double toDouble(string s) {
  /**
   * @brief Converts a string to a double.
   * @param s The string to convert.
   * @return double The double-precision floating-point representation of the string.
   */
  double r = 0;
  istringstream sin(s);
  sin >> r;
  return r;
}
// ENDTEMPLATE_BY_ZAYED

int M[1000][3];

int main() {
  /**
   * @brief Main function implementing the solution for the "Team" problem.
   *
   * Reads the number of problems (n) and for each problem, reads three integers
   * representing the certainty of three friends about a solution. It then counts
   * and prints the total number of problems that will be implemented, based on
   * the rule that at least two friends must be sure.
   * @return int Returns 0 upon successful execution.
   */
  // std::ios_base::sync_with_stdio(0);
  int cases, caseno = 0, n, I, K = 0, cnt = 0, temp = 0;
  scanf("%d", &n); // Reads the number of problems (n).

  // Block Logic: Reads the certainty values for each problem from input.
  // Pre-condition: 'n' specifies the number of problems.
  // Invariant: The M array stores the three certainty values for each of 'n' problems.
  for (I = 0; I < n; I++) {
    for (K = 0; K < 3; K++) // Loop to read three certainty values for the current problem.
      scanf("%d", &M[I][K]);
  }

  // Block Logic: Iterates through each problem to determine if it should be implemented.
  // Pre-condition: The M array is populated with certainty values.
  // Invariant: 'cnt' accumulates the count of problems where at least two friends are sure.
  for (I = 0; I < n; I++) {
    temp = 0; // Resets temporary counter for current problem.
    // Block Logic: Counts how many friends are sure for the current problem.
    // Pre-condition: M[I][K] values are 0 or 1.
    // Invariant: 'temp' stores the sum of certainty values for the current problem.
    for (K = 0; K < 3; K++) {
      if (M[I][K] == 1) // Checks if the current friend is sure.
        temp++;        // Increments if sure.
    }
    // Block Logic: Determines if the problem should be implemented.
    // Pre-condition: 'temp' holds the count of sure friends for the problem.
    // Post-condition: 'cnt' is incremented if at least two friends are sure.
    if (temp >= 2) // If at least two friends are sure (sum >= 2).
      cnt++;       // Increment the implemented problems counter.
  }
  printf("%d\n", cnt); // Prints the total count of implemented problems.
  return 0;
}
