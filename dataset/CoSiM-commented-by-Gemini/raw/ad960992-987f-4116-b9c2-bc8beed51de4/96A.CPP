/**
 * @file 96A.CPP
 * @brief Determines if a football team has a "dangerous" situation.
 *
 * This program reads a string representing a sequence of players from two
 * different teams (represented by '0' and '1'). It determines if there is a
 * continuous sequence of 7 or more players from the same team.
 *
 * @note The extensive use of typedefs and macros is a common practice in
 * competitive programming for speed and convenience.
 *
 * @see https://codeforces.com/problemset/problem/96/A
 */
/*
  Yadi Qian
*/

#include <bits/stdc++.h>

using namespace std;

const double pi = acos(-1.0);
typedef pair<int, int> PII;
typedef vector<int> VI;
typedef vector<string> VS;
typedef vector<PII> VII;
typedef vector<VI> VVI;
typedef map<int, int> MPII;
typedef set<int> SETI;
typedef multiset<int> MSETI;
typedef long int int32;
typedef unsigned long int uint32;
typedef long long int int64;
typedef unsigned long long int uint64;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  string a;
  cin >> a;
  int letter = a[0];
  int count = 0;
  /**
   * @brief Main processing loop.
   *
   * Iterates through the string of players to find a continuous sequence of
   * identical characters.
   */
  for (int i = 0; i < a.length() - 1; i++) {
    /**
     * @brief Sequence counting logic.
     *
     * If the current player is the same as the next one, the counter is
     * incremented. If they are different, the counter is reset.
     * A sequence of 7 is detected when the count of identical adjacent
     * characters reaches 6.
     */
    if (a[i] == a[i + 1])
      count++;
    else
      count = 0;

    if (count >= 6) {
      cout << "YES";
      return 0;
    }
  }

  cout << "NO";
  return 0;
}