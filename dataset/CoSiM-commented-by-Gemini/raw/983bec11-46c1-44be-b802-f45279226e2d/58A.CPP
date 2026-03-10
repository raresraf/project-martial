/**
 * @file 58A.CPP
 * @brief Solution for a competitive programming problem (Codeforces 58A - Chat room).
 * The program checks if the string "hello" is a subsequence of the given input string.
 *
 * NOTE: A significant portion of this file consists of a boilerplate template with
 *       various includes, macros, and utility functions, which is a common practice
 *       in competitive programming. The core logic for this specific problem
 *       is located at the bottom of the file.
 */
// Codeforces 58A - Chat room.cpp
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

#define NL '\n'
using namespace std;

// --- BEGIN BOILERPLATE TEMPLATE ---
// This section contains a large collection of pre-written utility code
// for various competitive programming tasks. It is not specific to this problem.
// BEGINTEMPLATE_BY_ZAYED
const double PI = 2 * acos(0.0);
const double EPS = 1e-11;
const int sz = 100;

typedef unsigned long long ull;    // NOTES:"%llu"
typedef long long ll;              // NOTES:"%lld"
typedef long long int64;           // NOTES:int64
typedef unsigned long long uint64; // NOTES:uint64

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

template <class T> inline void checkmin(T &a, T b) {
  if (b < a)
    a = b;
} // NOTES:checkmin(
template <class T> inline void checkmax(T &a, T b) {
  if (b > a)
    a = b;
} // NOTES:checkmax(
template <class T> inline T sqr(T x) { return x * x; } // NOTES:square
template <class T> inline T _max(T a, T b) { return (a > b) ? a : b; }
template <class T> inline T _min(T a, T b) { return (a < b) ? a : b; }
template <class T> inline T _pow(T base, T power) {
  T I, result = base;
  if (power == 0)
    return 1;
  for (I = 0; I < power - 1; I++)
    result = result * base;
  return result;
}
// Translator
bool isUpperCase(char c) { return c >= 'A' && c <= 'Z'; } // NOTES: isUpperCase(
bool isLowerCase(char c) { return c >= 'a' && c <= 'z'; } // NOTES: isLowerCase(
bool isLetter(char c) {
  return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z';
} // NOTES: isLetter(
bool isDigit(char c) { return c >= '0' && c <= '9'; } // NOTES:isDigit(
char toLowerCase(char c) {
  return (isUpperCase(c)) ? (c + 32) : c;
} // NOTES: toLowerCase(
char toUpperCase(char c) {
  return (isLowerCase(c)) ? (c - 32) : c;
} // NOTES: toUpperCase(
int toInt(string s) {
  int r = 0;
  istringstream sin(s);
  sin >> r;
  return r;
} // NOTES:toInt(
double toDouble(string s) {
  double r = 0;
  istringstream sin(s);
  sin >> r;
  return r;
} // NOTES:toDouble(
// ENDTEMPLATE_BY_ZAYED
// --- END BOILERPLATE TEMPLATE ---

char S[100 + 10];
char T[] = "hello";

/**
 * @brief Core logic for the "Chat room" problem.
 * @algorithm Greedy linear scan for subsequence checking.
 * @time_complexity O(L), where L is the length of the input string.
 * @space_complexity O(L) for storing the input string.
 */
int main() {
  std::ios_base::sync_with_stdio(0); // Fast I/O
  int tcases, caseno = 1, n, I, K = 0, cnt;
  scanf("%s", S);
  int len = strlen(S);
  bool flag = false;

  /**
   * @block
   * @brief Scans the input string `S` to find the subsequence "hello".
   *
   * Invariant: `K` is the index of the next character to find in the target string `T` ("hello").
   * The loop greedily matches characters. When a character `S[I]` matches `T[K]`,
   * `K` is incremented to look for the next character in "hello".
   */
  for (I = 0; I < len; I++) {
    if (S[I] == T[K])
      K++;
    // If K reaches 5, all characters of "hello" have been found in order.
    if (K == 5) {
      flag = true;
      break;
    }
  }
  if (flag)
    printf("YES\n");
  else
    printf("NO\n");

  return 0;
}
