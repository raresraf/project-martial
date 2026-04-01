/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem from Codeforces (Problem 112A).
 * @details This program compares two strings lexicographically, ignoring case.
 * It outputs -1 if the first string is lexicographically smaller than the second,
 * 1 if it is larger, and 0 if they are equal.
 *
 * Algorithm: Case-insensitive string comparison.
 * 1. Read two strings from standard input.
 * 2. Convert both strings to the same case (lowercase).
 * 3. Perform a standard lexicographical comparison on the modified strings.
 *
 * Time Complexity: O(L) where L is the length of the strings, for conversion and comparison.
 * Space Complexity: O(L) to store the strings.
 */
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <utility>
#include <vector>

// using
using namespace std;

// typedef
typedef vector<int> VI;
typedef vector<VI> VII;
typedef vector<string> VS;
typedef vector<long> VL;
typedef long long LL;
typedef vector<LL> VLL;

// container utils
#define PB push_back
#define PF push_front
#define GRT(x) greater<(x)>()
#define ASORT(x) sort((x).begin(), (x).end())
#define DSORT(x, y) sort((x).begin(), (x).end(), greater<(y)>())
#define FILL(x, y) fill((x).begin(), (x).end(), (y))
#define COPY(x, y)                                                             \
  (y).clear();                                                                 \
  copy((x).begin(), (x).end(), back_inserter(y))

// repetition
#define FOR(i, a, b) for (int i = (a); i < (b); i++)
#define FORL(i, a, b) for (long i = (a); i < (b); i++)
#define REP(i, n) FOR(i, 0, n)
#define REPL(i, n) FORL(i, 0, n)

// debug
#define DUMP(a) cerr << #a << " = " << (a) << endl
#define DUMP2(a, b)                                                            \
  cerr << #a << " = " << (a) << ", " << #b << " = " << (b) << endl
#define DUMP3(a, b, c)                                                         \
  cerr << #a << " = " << (a) << ", " << #b << " = " << (b) << ", " << #c       \
       << " = " << (c) << endl

/**
 * @brief Converts a string to lowercase.
 * @param str The input string.
 * @return The lowercase version of the string.
 */
string toLower(string str) {
  REP(i, str.size()) {
    if (str[i] >= 'A' && str[i] <= 'Z') {
      str[i] += ('a' - 'A');
    }
  }

  return str;
}

/**
 * @brief Main logic for solving the problem.
 * Reads two strings, converts them to lowercase, and compares them.
 */
void solve() {
  string s, t;

  cin >> s >> t;

  s = toLower(s);
  t = toLower(t);

  // Lexicographically compares the two lowercase strings.
  if (s < t)
    cout << -1 << endl;
  else if (s == t)
    cout << 0 << endl;
  else
    cout << 1 << endl;
}

int main(int argc, char *argv[]) {
  solve();

  return 0;
}
