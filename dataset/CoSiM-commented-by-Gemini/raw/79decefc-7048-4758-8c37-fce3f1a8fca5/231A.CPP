/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A, "Team".
 * @details This program calculates the number of problems a team will implement.
 * A problem is implemented if at least two of the three team members are sure
 * about its solution. The file includes a large, general-purpose template for
 * competitive programming, most of which is not used in this specific solution.
 *
 * Algorithm:
 * 1. Read the total number of problems, `n`.
 * 2. Read all `n` lines of input, where each line contains the three members'
 *    opinions (1 for sure, 0 for not sure), and store them in a 2D array `M`.
 * 3. Iterate through each row of the array `M`, representing each problem.
 * 4. For each problem, count the number of members who are sure (i.e., the count of 1s).
 * 5. If this count is 2 or greater, increment a total problem counter.
 * 6. After checking all problems, print the final total.
 *
 * Time Complexity: O(N) because it iterates through the N*3 inputs twice in
 * separate loops.
 * Space Complexity: O(N) to store all inputs in the 2D array `M`.
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

#define LL long long
#define MAX 100005
#define MOD 1000000007

#define NL '\n'
using namespace std;

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

int M[1000][3];

int main() {
  // std::ios_base::sync_with_stdio(0);
  int cases, caseno = 0, n, I, K = 0, cnt = 0, temp = 0;
  scanf("%d", &n);

  /**
   * @brief Reads all problem data from standard input and stores it in a 2D
   * array. This approach reads all data before processing begins.
   */
  for (I = 0; I < n; I++) {
    for (K = 0; K < 3; K++)
      scanf("%d", &M[I][K]);
  }

  /**
   * @brief Iterates through each problem stored in the array to determine if
   * the team will solve it.
   */
  for (I = 0; I < n; I++) {
    temp = 0;
    // Inner loop to sum the opinions for the current problem.
    for (K = 0; K < 3; K++) {
      if (M[I][K] == 1)
        temp++;
    }
    /**
     * @brief If two or more members are sure, increment the counter for
     * problems to be implemented.
     */
    if (temp >= 2)
      cnt++;
  }
  printf("%d\n", cnt);
  return 0;
}
