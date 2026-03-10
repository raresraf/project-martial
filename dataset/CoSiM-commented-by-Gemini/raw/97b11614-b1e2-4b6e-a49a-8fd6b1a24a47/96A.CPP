/**
 * @file 96A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Football").
 * This program determines if a given string contains a contiguous sequence of
 * at least 7 identical characters.
 *
 * @algorithm Single-pass linear scan.
 * @time_complexity O(N), where N is the length of the input string, as it iterates through the string once.
 * @space_complexity O(N) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int c = 1, i, k = 0;
  string s;
  cin >> s;

  /**
   * @block
   * @brief Iterates through the string to find a run of 7 identical characters.
   *
   * Invariant: `c` holds the length of the current contiguous run of identical
   * characters ending at index `i`.
   *
   * The loop compares each character with the one preceding it. If they are the
   * same, the run counter `c` is incremented. If `c` reaches 7, a flag `k` is
   * set, "YES" is printed, and the loop terminates. If the characters are
   * different, the run is broken, and `c` is reset to 1.
   */
  for (i = 1; i < s.length(); i++) {
    if (s[i] == s[i - 1]) {
      c++;

      if (c == 7) {
        cout << "YES" << endl;
        k = 1; // Flag to indicate a successful find
        break;
      }
    } else if (s[i] != s[i - 1])
      c = 1; // Reset counter for a new run
  }

  // If the loop completes without the flag `k` being set, no run of 7 was found.
  if (k == 0) {
    cout << "NO" << endl;
  }
  return 0;
}
