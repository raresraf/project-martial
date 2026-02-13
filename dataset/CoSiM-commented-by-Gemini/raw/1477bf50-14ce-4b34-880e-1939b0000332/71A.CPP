/**
 * @file 71A.CPP
 * @brief Solution for the "Way Too Long Words" problem.
 * @details This program processes a list of words. If a word's length is greater than 10 characters,
 * it is abbreviated. The abbreviation consists of the first character, the number of characters
 * between the first and last (i.e., length - 2), and the last character. Words with 10 or fewer
 * characters are printed as is. This file includes a large boilerplate template common in
 * competitive programming.
 *
 * @section algorithm Algorithm
 * 1. Read the number of test cases.
 * 2. Loop for each test case:
 *    a. Read the entire line containing the word into a C-style string.
 *    b. Calculate the length of the string.
 *    c. If the length is greater than 10:
 *       - Print the first character, the length minus 2, and the last character, followed by a newline.
 *    d. Else (if the length is 10 or less):
 *       - Print the string as is, followed by a newline.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(T * L), where T is the number of test cases and L is the maximum length of a word. The complexity is dominated by input/output operations.
 * - Space Complexity: O(L), for the character array used to store the word.
 */
// Codeforces 71A - Way Too Long Words.cpp
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

// NOTE: The following is a standard template of helper functions and type definitions
// often used in competitive programming to accelerate development.
// BEGINTEMPLATE_BY_ZAYED
const double PI = 2 * acos(0.0);
const double EPS = 1e-11;
const int sz = 1e5 + 10;

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
int _max(int a, int b) { return (a > b) ? a : b; }
int _min(int a, int b) { return (a < b) ? a : b; }
inline int _pow(int base, int power) {
  int I, result = base;
  if (power == 0)
    return 1;
  for (I = 0; I < power - 1; I++) {
    result = result * base;
  }
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

char s[110];
int main() {
  // std::ios_base::sync_with_stdio(0);
  int cases, caseno = 0, n, I, K;
  char iFalse;
  // Read the number of test cases. The '%c' is used to consume the newline character
  // left by scanf, which is important for the subsequent getline call.
  scanf("%d%c", &cases, &iFalse);
  while (cases--) {
    // Read the entire line into the character array 's'.
    cin.getline(s, 110);
    int len = strlen(s);

    // If the word's length is 10 or less, print it as is.
    if (len <= 10)
      printf("%s\n", s);
    // Otherwise, print the abbreviated version.
    else
      printf("%c%d%c\n", s[0], len - 2, s[len - 1]);
  }

  return 0;
}
