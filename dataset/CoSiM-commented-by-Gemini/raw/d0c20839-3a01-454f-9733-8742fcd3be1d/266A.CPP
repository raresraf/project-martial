/**
 * @file 266A.CPP
 * @brief Counts adjacent identical stones to determine minimum removals.
 *
 * This program solves the "Stones on the Table" problem. The goal is to find the
 * minimum number of stones to remove from a row so that no two adjacent stones
 * share the same color. This is equivalent to counting every pair of adjacent,
 * identical stones.
 *
 * @b Algorithm: Linear Scan
 *   The code iterates through the sequence of stones and increments a counter
 *   each time it finds a stone that is identical to the one immediately
 *   following it.
 *
 * @b Time_Complexity: O(N), where N is the number of stones.
 * @b Space_Complexity: O(N) to store the sequence of stones.
 *
 * @note The code declares many variables that are never used (e.g., m, j, q, b).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: number of stones (read but not used directly in the loop).
  // f: counter for adjacent identical stones.
  // The other integer variables are unused.
  int n, m, i, j, q, y, aa, bb, f = 0, s = 0, k, min = 1000000010, r, t, u = -1;
  // a: stores the sequence of stones. b: unused.
  char a[10010], b[10010];
  ;

  // Read the number of stones and the sequence.
  cin >> n >> a;
  // r: length of the stone sequence string.
  r = strlen(a);

  /**
   * @brief Iterate through the stones and count adjacent pairs.
   */
  for (i = 0; i < r; i++) {
    // If a stone is the same as the one next to it, increment the counter.
    if (a[i] == a[i + 1])
      f++;
  }

  // The final count `f` is the minimum number of stones to remove.
  cout << f << endl;

  return 0;
}