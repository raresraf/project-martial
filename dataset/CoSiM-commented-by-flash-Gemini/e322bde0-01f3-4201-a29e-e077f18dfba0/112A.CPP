/**
 * @file 112A.CPP
 * @brief This program solves the "Petya and Strings" problem from Codeforces.
 *        It compares two input strings lexicographically, ignoring case.
 *        The comparison result indicates which string comes first, or if they are equal.
 *
 * Algorithm:
 * 1. Read two input strings.
 * 2. Convert both strings to a consistent case (e.g., uppercase) for case-insensitive comparison.
 * 3. Perform a standard lexicographical comparison on the case-converted strings.
 * 4. Output -1 if the first string is lexicographically smaller, 1 if it's larger, and 0 if they are equal.
 *
 * Time Complexity: O(L), where L is the length of the strings, due to single passes for case conversion and comparison.
 * Space Complexity: O(L) for storing the two strings, plus additional space for case-converted copies (though in-place modification is used here).
 */
// Codeforces 112A - Petya and Strings.cpp
#include <cstdio>     // Standard C input/output.
#include <iostream>   // Standard C++ input/output.

#include <cstring>    // C-style string functions.
#include <sstream>    // String stream operations.
#include <string>     // C++ string class.

#include <deque>      // Double-ended queue container.
#include <list>       // Doubly-linked list container.
#include <map>        // Associative array/map container.
#include <queue>      // Queue container adapter.
#include <set>        // Set container.
#include <stack>      // Stack container adapter.
#include <vector>     // Dynamic array/vector container.

#include <algorithm>  // Standard algorithms like sort, min, max.
#include <bitset>     // Fixed-size sequence of bits.
#include <functional> // Function objects.
#include <numeric>    // Numeric operations.
#include <utility>    // Utility classes like pair.

#include <cctype>     // Character classification and conversion functions (e.g., toupper, isupper).
#include <cmath>      // Mathematical functions.
#include <cstdlib>    // General utilities.
#include <ctime>      // Time and date functions.
#include <iomanip>    // Input/output manipulators.

// Macros for type aliases and common constants.
#define LL long long     // Alias for long long.
#define MAX 100005       // Maximum value constant.
#define MOD 1000000007   // Modulo constant.

#define NL '\n'          // Alias for newline character, often used for faster output.
using namespace std;     // Uses the standard namespace to avoid repetitive std:: prefix.

// BEGINTEMPLATE_BY_ZAYED - A collection of common competitive programming utilities.
const double PI = 2 * acos(0.0); // Mathematical constant PI.
const double EPS = 1e-11;        // Epsilon for floating-point comparisons.
const int sz = 100;              // Size constant.

// Type aliases for various data structures and types.
typedef unsigned long long ull;    // Unsigned long long.
typedef long long ll;              // Long long.
typedef long long int64;           // Another alias for long long.
typedef unsigned long long uint64; // Another alias for unsigned long long.

typedef stack<int> si;       // Stack of integers.
typedef queue<int> qi;       // Queue of integers.
typedef vector<int> vi;      // Vector of integers.
typedef pair<ll, ll> pll;    // Pair of long longs.
typedef map<int, int> mii;   // Map from int to int.
typedef pair<int, int> pii;  // Pair of integers.
typedef map<char, int> mci;  // Map from char to int.
typedef map<string, int> msi; // Map from string to int.
typedef pair<string, int> psi; // Pair of string and int.
typedef map<string, string> mss; // Map from string to string.

// Generic utility functions.
template <class T> inline void checkmin(T &a, T b) {
  if (b < a)
    a = b;
} // Function to find and assign minimum.
template <class T> inline void checkmax(T &a, T b) {
  if (b > a)
    a = b;
} // Function to find and assign maximum.
template <class T> inline T sqr(T x) { return x * x; } // Square a number.
template <class T> inline T _max(T a, T b) { return (a > b) ? a : b; } // Custom max function.
template <class T> inline T _min(T a, T b) { return (a < b) ? a : b; } // Custom min function.
template <class T> inline T _pow(T base, T power) {
  T I, result = base;
  if (power == 0)
    return 1;
  for (I = 0; I < power - 1; I++)
    result = result * base;
  return result;
} // Custom power function.
// Translator functions (character conversions and checks).
bool isUpperCase(char c) { return c >= 'A' && c <= 'Z'; } // Checks if char is uppercase.
bool isLowerCase(char c) { return c >= 'a' && c <= 'z'; } // Checks if char is lowercase.
bool isVowel(char c) {
  return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' ||
          c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U');
} // Checks if char is a vowel.
bool isLetter(char c) {
  return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z';
} // Checks if char is a letter.
bool isDigit(char c) { return c >= '0' && c <= '9'; } // Checks if char is a digit.
char toLowerCase(char c) {
  return (isUpperCase(c)) ? (c + 32) : c;
} // Converts char to lowercase.
char toUpperCase(char c) {
  return (isLowerCase(c)) ? (c - 32) : c;
} // Converts char to uppercase.
int toInt(string s) {
  int r = 0;
  istringstream sin(s);
  sin >> r;
  return r;
} // Converts string to int.
double toDouble(string s) {
  double r = 0;
  istringstream sin(s);
  sin >> r;
  return r;
} // Converts string to double.
// ENDTEMPLATE_BY_ZAYED

string S, T; // Global string variables for input.

/**
 * @brief Main function: Reads two strings, converts them to uppercase,
 *        and compares them lexicographically.
 * Functional Utility: Entry point of the program, orchestrates the comparison logic.
 */
int main() {
  // Functional Utility: Disables synchronization between C-style I/O and C++ streams
  // for faster input/output operations, common in competitive programming.
  std::ios_base::sync_with_stdio(0);
  // Various unused variables (tcases, caseno, n, K, cnt, sum1, sum2) common in competitive programming templates.
  int tcases, caseno = 1, n, I, K = 0, cnt, len, sum1 = 0, sum2 = 0;

  // Reads the first input string into global variable S.
  cin >> S;
  // Reads the second input string into global variable T.
  cin >> T;
  // Gets the length of string S (assuming S and T have equal length as per problem constraints).
  len = S.length();

  /**
   * Block Logic: Converts both strings S and T to uppercase in-place.
   * This ensures a case-insensitive comparison in the subsequent step.
   * Uses the custom `toUpperCase` helper function from the template.
   * Invariant: After this loop, both S and T contain only uppercase characters.
   */
  for (I = 0; I < len; I++) {
    S[I] = toUpperCase(S[I]);
    T[I] = toUpperCase(T[I]);
  }
  // Prints the result of the lexicographical comparison of the two uppercase strings.
  // std::string::compare returns:
  //   - A negative value if S is lexicographically smaller than T.
  //   - A positive value if S is lexicographically larger than T.
  //   - Zero if S and T are equal.
  cout << (S.compare(T)) << NL;
  return 0; // Standard program exit.
}
