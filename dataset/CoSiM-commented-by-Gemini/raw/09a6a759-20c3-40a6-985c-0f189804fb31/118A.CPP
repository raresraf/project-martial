/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem using a large competitive programming template.
 * @details This file contains a solution to the Codeforces "String Task" problem. The core logic
 * is embedded within a large, reusable template of macros and helper functions, which is a
 * common practice in competitive programming. The program's purpose is to take a string,
 * remove all vowels, insert a '.' before each consonant, and convert all consonants to lowercase.
 *
 * @see https://codeforces.com/problemset/problem/118/A
 *
 * Algorithm:
 * 1. The input string is read.
 * 2. The code iterates through the string, character by character.
 * 3. For each character, a custom helper function, `isVowel`, is called.
 * 4. If the character is not a vowel, a '.' is printed to standard output.
 * 5. The character is then converted to lowercase using another custom helper, `toLowerCase`,
 *    and printed.
 *
 * Time Complexity: O(N), where N is the length of the string, due to a single pass.
 * Space Complexity: O(N) to store the input string.
 */

// Includes a wide range of standard library headers, typical for competitive programming.
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

// --- Start of a large, reusable competitive programming template ---

// Macros for shortening common code patterns.
#define NL '\n'
#define FOR(I, J, K) for (I = J; I < K; I++)
#define REV(I, J, K) for (I = J; I > K; I--)
#define sf1(x) scanf("%d", &x)
#define sf scanf
#define pf1(x) printf("%d", x)
#define pf printf
#define PC() printf("Case %d: ", caseno++)
#define READ() freopen("in.txt", "r", stdin)
#define WRITE() freopen("out.txt", "w", stdout)
using namespace std;

// Typedefs for common data types.
typedef unsigned long long ULL;
typedef unsigned long long uint64;
typedef long long LL;
typedef long long int64;

// Generic template functions for math and utility operations.
template <class T> inline void checkmax(T &a, T b) { if (b > a) a = b; }
template <class T> inline void checkmin(T &a, T b) { if (b < a) a = b; }
template <class T> inline T MAX(T a, T b) { return (a > b) ? a : b; }
template <class T> inline T MIN(T a, T b) { return (a < b) ? a : b; }
template <class T> inline T sqr(T x) { return x * x; }
template <class T> inline T fABS(T a) { return a < 0 ? a * (-1) : a; }
template <class T> inline void SWAP(T &a, T &b) { T t = a; a = b; b = t; }
inline LL POW(LL base, LL power) { /* ... implementation ... */ }

// Custom implementations of character functions, often replicating <cctype> functionality.
bool isUpperCase(char c) { return c >= 'A' && c <= 'Z'; }
bool isLowerCase(char c) { return c >= 'a' && c <= 'z'; }
bool isVowel(char c) {
  return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' ||
          c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' ||
          c == 'y' || c == 'Y');
}
bool isLetter(char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
bool isDigit(char c) { return c >= '0' && c <= '9'; }
char toLowerCase(char c) { return (isUpperCase(c)) ? (c + 32) : c; }
char toUpperCase(char c) { return (isLowerCase(c)) ? (c - 32) : c; }

// Custom string-to-number conversion functions.
int toInt(string s) { /* ... implementation ... */ }
double toDouble(string s) { /* ... implementation ... */ }
string toString(int n) { /* ... implementation ... */ }

// --- End of the template ---

#define INF 2147483647
#define MOD 1000000007
const double PI = 2 * acos(0.0);
const double EPS = 1e-11;
const int SIZE = 1e6;

string S;

int main() {
  // Optimizes C++ I/O by decoupling from C stdio.
  std::ios_base::sync_with_stdio(0);

  int tcases, I, J, K, N, n, m, cnt = 0, len;
  cin >> S;
  len = S.length();

  /**
   * @brief Main processing loop.
   * Iterates through the input string, applying the transformations.
   */
  for (I = 0; I < len; I++) {
    // Check if the character is a vowel using the custom helper function.
    if (!isVowel(S[I])) {
      cout << ".";
      // Print the consonant, ensuring it is lowercase.
      if (isLowerCase(S[I]))
        cout << S[I];
      else
        cout << toLowerCase(S[I]);
    }
  }
  cout << NL;
  return 0;
}
