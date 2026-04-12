/**
 * @file 339A.CPP
 * @brief Logic for maximizing pattern formation from a constrained character pool.
 * 
 * Algorithm: Resource bottleneck analysis for the word "nineteen".
 * Logic: Calculates the maximum number of times the word "nineteen" can be formed 
 * using available characters {'n', 'i', 'e', 't'}, accounting for overlapping 'n' 
 * characters in sequential formations (n-i-n-e-t-e-e-n-i-n-e-t-e-e-n...).
 * 
 * Resource Requirements per formation (K):
 * - 'n': 2K + 1 (overlapping start/end)
 * - 'i': K
 * - 'e': 3K
 * - 't': K
 * 
 * Time Complexity: $O(N)$ where N is the length of the input string.
 * Space Complexity: $O(N)$ for the input buffer.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  string str;
  // Counters for the required character subset.
  int i = 0, n = 0, t = 0, e = 0;
  cin >> str;

  /**
   * Block Logic: Frequency analysis of the input character stream.
   * Invariant: Counts occurrences of characters critical to the "nineteen" pattern.
   */
  for (int j = 0; j < str.size(); j++) {
    if (str[j] == 'n')
      n++;
    else if (str[j] == 't')
      t++;
    else if (str[j] == 'e')
      e++;
    else if (str[j] == 'i')
      i++;
  }

  /**
   * Functional Utility: Normalizes counts based on specific character stoichiometry.
   * Logic: 
   * 1. 'n' requires 3 for the first and 2 for each subsequent (Overlap logic: (n-1)/2).
   * 2. 'e' requires exactly 3 per word.
   * 3. 'i' and 't' require exactly 1 per word.
   */
  n--;      // Initial 'n' offset for the first word.
  e /= 3;   // 'e' bottleneck.
  n /= 2;   // 'n' bottleneck.

  /**
   * Functional Utility: Identifies the limiting resource (bottleneck) across all components.
   */
  cout << min(n, min(e, min(i, t)));
  
  return 0;
}
