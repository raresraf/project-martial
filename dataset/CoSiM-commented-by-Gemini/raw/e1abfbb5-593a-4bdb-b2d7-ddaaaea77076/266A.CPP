/**
 * @file 266A.CPP
 * @brief Solution to the "Stones on the Table" problem.
 * @see https://codeforces.com/problemset/problem/266/A
 *
 * This program calculates the minimum number of stones to remove from a row
 * so that no two adjacent stones have the same color.
 *
 * Algorithm:
 * 1. Read the number of stones (n) and the string representing the colors of
 *    the stones.
 * 2. Iterate through the string, comparing each stone with the one next to it.
 * 3. If two adjacent stones have the same color, increment a counter.
 * 4. The final value of the counter is the minimum number of stones to remove.
 *
 * Time Complexity: O(N), where N is the number of stones.
 * Space Complexity: O(N) to store the input string.
 */
// https://codeforces.com/problemset/problem/266/A

#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Solves the "Stones on the Table" problem.
 * @param s A string representing the colors of the stones.
 * @return The minimum number of stones to remove.
 */
int solve(const string &s) {
  if (s.length() == 0)
    return 0;
  int cnt = 0;
  // Invariant: Iterates through the string, counting adjacent stones of the
  // same color.
  for (int i = 1; i < s.length(); ++i) {
    // Pre-condition: If a stone has the same color as the one before it, it
    // needs to be removed.
    if (s[i - 1] == s[i]) {
      cnt++;
    }
  }

  return cnt;
}

int main() {
  int n;
  cin >> n;
  string s;
  cin >> s;
  cout << solve(s) << endl;
}