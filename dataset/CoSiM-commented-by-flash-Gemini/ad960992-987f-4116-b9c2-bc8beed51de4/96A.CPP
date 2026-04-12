/**
 * @file 96A.CPP
 * @brief Logic for detecting hazardous clustering in a binary state sequence.
 * 
 * Logic: Identifies "dangerous" states in a team formation (represented as a binary 
 * string of '0's and '1's). A situation is dangerous (YES) if there is a contiguous 
 * segment of at least 7 players from the same team.
 * 
 * Algorithm: Single-pass linear scan with an adjacency-based counter.
 * Time Complexity: $O(N)$ where N is the length of the formation string.
 * Space Complexity: $O(N)$ for the input buffer.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  /**
   * Functional Utility: Optimizes I/O performance by decoupling C++ streams from C standard streams.
   */
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  string a;
  // Reads the binary formation sequence.
  cin >> a;
  
  // count: Tracks consecutive identical characters.
  int count = 0;

  /**
   * Block Logic: Scans the sequence for contiguous identical segments.
   * Invariant: 'count' represents the number of consecutive elements matching 
   * the team of the player at index i.
   */
  for (int i = 0; i < (int)a.length() - 1; i++) {
    /**
     * Case 1: Sequence continues with the same team.
     */
    if (a[i] == a[i + 1])
      count++;
    else
      /**
       * Case 2: Team transition detected.
       * Logic: Resets the counter for the new team block.
       */
      count = 0;

    /**
     * Functional Utility: Checks for the danger threshold.
     * Logic: Since count starts at 0 for 1 player, a count of 6 implies 
     * a contiguous block of 7 players.
     */
    if (count >= 6) {
      cout << "YES";
      return 0;
    }
  }

  // Output "NO" if no hazardous clustering is identified.
  cout << "NO";
  
  return 0;
}
