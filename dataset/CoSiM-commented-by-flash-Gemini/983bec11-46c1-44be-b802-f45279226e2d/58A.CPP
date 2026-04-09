/**
 * @file 58A.CPP
 * @brief This program checks if the word "hello" can be formed as a subsequence
 * of a given input string. That is, if all characters of "hello" appear in the
 * input string in the correct order, but not necessarily contiguously.
 *
 * Algorithm: Subsequence matching (two-pointer approach).
 * 1. Initialize a pointer for the target word "hello" and another for the input string.
 * 2. Iterate through the input string, advancing the target word pointer whenever a match is found.
 * 3. If all characters of "hello" are matched, the word is a subsequence.
 * Time Complexity: O(L) where L is the length of the input string, as it involves a single pass.
 * Space Complexity: O(1) for storing pointers and a flag, excluding input/output strings.
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

#define NL '
'
using namespace std;

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

// Functional Utility: Character array to store the input string.
char S[100 + 10];
// Functional Utility: Character array representing the target subsequence "hello".
char T[] = "hello";

/**
 * @brief Main function to solve the "Chat room" problem.
 * Reads an input string and determines if "hello" is a subsequence.
 * @return 0 upon successful execution.
 */
int main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output.
  std::ios_base::sync_with_stdio(0);
  // Functional Utility: Declares variables. 'K' acts as a pointer for the target string 'T'.
  int tcases, caseno = 1, n, I, K = 0, cnt;
  // Functional Utility: Reads the input string into character array 'S'.
  scanf("%s", S);
  // Functional Utility: Calculates the length of the input string 'S'.
  int len = strlen(S);
  // Functional Utility: Boolean flag to indicate if "hello" has been found as a subsequence.
  bool flag = false;

  // Block Logic: Iterates through the input string 'S' to find characters of 'T' ("hello") in sequence.
  // Invariant: 'K' tracks the number of characters of 'T' found so far in 'S' in correct order.
  for (I = 0; I < len; I++) {
    // Block Logic: If the current character in 'S' matches the next expected character in 'T',
    // advance the 'T' pointer.
    if (S[I] == T[K])
      K++;
    // Block Logic: If all characters of 'T' have been found (K reaches 5), set the flag and break.
    // Invariant: If K becomes 5, "hello" is confirmed as a subsequence.
    if (K == 5) {
      flag = true;
      break;
    }
  }
  // Block Logic: Prints "YES" or "NO" based on whether "hello" was found as a subsequence.
  if (flag)
    printf("YES
");
  else
    printf("NO
");

  return 0;
}
