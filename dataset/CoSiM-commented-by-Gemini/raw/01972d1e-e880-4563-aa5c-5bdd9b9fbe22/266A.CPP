/**
 * @file 266A.CPP
 * @brief Counts the minimum number of stones to remove from a row to avoid consecutive identical colors.
 *
 * This program solves a problem where stones of different colors are arranged in a row.
 * The goal is to find the minimum number of stones to remove so that no two adjacent
 * stones have the same color. This is equivalent to counting the number of pairs of
 * adjacent identical stones.
 *
 * @b Algorithm: The program iterates through the sequence of stones from the second stone
 * onwards. It compares each stone with the one immediately preceding it. If they have the
 * same color, a counter is incremented. The final value of this counter is the answer.
 *
 * @b Time_Complexity: O(N), where N is the number of stones, due to the single pass through the input string.
 * @b Space_Complexity: O(N) to store the string representing the colors of the stones.
 */
/* Monil Soni sonil06 */
/* DA-IICT  */

#include <bits/stdc++.h>

using namespace std;

typedef long long LL;
typedef vector<int> vi;
typedef vector<long long> vl;

#define MOD 1000000007
#define _for(i, n) for (int i = 0; i < n; i++)
#define __for(i, n) for (int i = 1; i <= n; i++)
#define REP(i, s, n) for (int i = (s), _n = (n); i <= _n; i++)
#define FOR(i, n) for (int i = 0, _n = (n); i < _n; i++)
#define REPD(i, e, s) for (int i = (e), _s = (s); i >= _s; i--)
#define tr(container, it)                                                      \
  for (typeof(container.begin()) it = container.begin(); \
       it != container.end(); it++)
#define pb push_back
#define mp make_pair
#define fi first
#define se second
#define ALL(x) x.begin(), x.end()

int main() {
  int n;
  // Read the number of stones.
  scanf("%d", &n);
  string str;
  // Read the string of stone colors.
  cin >> str;
  int count = 0;
  /**
   * @brief This loop iterates through the stones to find adjacent pairs of the same color.
   *
   * Pre-condition: `str` contains the sequence of `n` stone colors.
   * Invariant: `count` holds the number of removals needed for the scanned part of the string.
   */
  for (int i = 1; i < n; i++) {
    // If a stone's color is the same as the previous one, increment the removal count.
    if (str.at(i - 1) == str.at(i))
      count++;
  }
  // Print the total number of stones to remove.
  printf("%d\n", count);
  return 0;
}