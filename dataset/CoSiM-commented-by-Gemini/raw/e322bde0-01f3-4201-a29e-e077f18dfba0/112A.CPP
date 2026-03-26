/**
 * @file 112A.CPP
 * @brief Solution to Codeforces problem 112A - Petya and Strings, using a large competitive programming template.
 * @description This file contains a solution to the "Petya and Strings" problem, which
 *              requires a case-insensitive lexicographical comparison of two strings.
 *              The majority of the file consists of a generic, reusable template for
 *              competitive programming, including numerous headers, type aliases, and
 *              utility functions. The actual problem-specific logic is located at the
 *              end of the `main` function.
 *
 * Algorithm: The core logic, separate from the template, is as follows:
 * 1. Read two strings, S and T.
 * 2. Iterate through both strings, converting each character to uppercase using a
 *    helper function provided by the template.
 * 3. Use the `string::compare` method to perform the lexicographical comparison.
 * 4. Print the result of the comparison. The `compare` method returns a negative,
 *    zero, or positive value, which aligns with the problem's output requirements
 *    (-1, 0, or 1, although any value with the correct sign is typically accepted).
 *
 * Time Complexity: O(N), where N is the length of the strings.
 * Space Complexity: O(N) to store the two input strings.
 */

// --- GENERIC COMPETITIVE PROGRAMMING TEMPLATE (BY ZAYED) ---
// This template includes a vast collection of headers, macros, type definitions,
// and utility functions intended to speed up development in programming contests.
// Most of this code is not used for solving this specific problem.
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

#define LL long long
#define MAX 100005
#define MOD 1000000007
#define NL '
'
using namespace std;

// BEGINTEMPLATE_BY_ZAYED
const double PI = 2 * acos(0.0);
const double EPS = 1e-11;
const int sz = 100;

typedef unsigned long long ull;
typedef long long ll;
typedef long long int64;
typedef unsigned long long uint64;
typedef stack<int> si;
typedef queue<int> qi;
typedef vector<int> vi;
typedef pair<ll, ll> pll;
typedef map<int, int> mii;
typedef pair<int, int> pii;
typedef map<char, int> mci;
typedef map<string, int> msi;
typedef pair<string, int> psi;
typedef map<string, string> mss;

template <class T> inline void checkmin(T &a, T b) { if (b < a) a = b; }
template <class T> inline void checkmax(T &a, T b) { if (b > a) a = b; }
template <class T> inline T sqr(T x) { return x * x; }
template <class T> inline T _max(T a, T b) { return (a > b) ? a : b; }
template <class T> inline T _min(T a, T b) { return (a < b) ? a : b; }
template <class T> inline T _pow(T base, T power) {
  T I, result = base;
  if (power == 0) return 1;
  for (I = 0; I < power - 1; I++) result = result * base;
  return result;
}
bool isUpperCase(char c) { return c >= 'A' && c <= 'Z'; }
bool isLowerCase(char c) { return c >= 'a' && c <= 'z'; }
bool isVowel(char c) {
  return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' ||
          c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U');
}
bool isLetter(char c) { return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z'; }
bool isDigit(char c) { return c >= '0' && c <= '9'; }
char toLowerCase(char c) { return (isUpperCase(c)) ? (c + 32) : c; }
char toUpperCase(char c) { return (isLowerCase(c)) ? (c - 32) : c; }
int toInt(string s) {
  int r = 0;
  istringstream sin(s);
  sin >> r;
  return r;
}
double toDouble(string s) {
  double r = 0;
  istringstream sin(s);
  sin >> r;
  return r;
}
// --- END OF TEMPLATE ---

string S, T;

int main() {
  // Speeds up C++ I/O operations.
  std::ios_base::sync_with_stdio(0);

  // Unused variables from the template.
  int tcases, caseno = 1, n, I, K = 0, cnt, len, sum1 = 0, sum2 = 0;

  cin >> S;
  cin >> T;
  len = S.length();

  /**
   * Block Logic: Normalize both strings to uppercase for case-insensitive comparison.
   */
  for (I = 0; I < len; I++) {
    S[I] = toUpperCase(S[I]);
    T[I] = toUpperCase(T[I]);
  }

  // Perform comparison and print the result. string::compare returns <0, 0, or >0.
  cout << (S.compare(T)) << NL;
  
  return 0;
}
