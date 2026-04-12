/**
 * @file 112A.CPP
 * @brief Case-insensitive lexicographical comparison of two strings using STL utilities.
 * 
 * Logic: Normalizes two input strings to lowercase and performs a character-by-character 
 * comparison to determine their relative dictionary order. Returns -1, 0, or 1.
 * 
 * Algorithm: Automated state normalization via std::transform followed by linear scan.
 * Time Complexity: $O(N)$ where N is the length of the strings.
 * Space Complexity: $O(N)$ for string storage.
 */

#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s1, s2;
  // Functional Utility: Reads two character sequences from standard input.
  if (!(cin >> s1 >> s2)) return 0;

  /**
   * Functional Utility: State normalization.
   * Logic: Uses std::transform to map all characters in both strings to their 
   * lowercase equivalents, ensuring case-insensitivity for subsequent comparisons.
   */
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);

  // State: Tracks if a terminating difference has been identified.
  bool difference_found = false;

  /**
   * Block Logic: Lexicographical order evaluation.
   * Invariant: Scans parallel indices until a character mismatch occurs.
   */
  for (int i = 0; i < (int)s1.length(); i++) {
    if (s1[i] < s2[i]) {
      // Logic: s1 precedes s2 in dictionary order.
      cout << "-1";
      difference_found = true;
      break;
    } else if (s1[i] == s2[i]) {
      // Logic: Current character pair is identical; continue scan.
      continue;
    } else {
      // Logic: s1 follows s2 in dictionary order.
      cout << "1";
      difference_found = true;
      break;
    }
  }

  /**
   * Functional Utility: Terminal equivalence check.
   * Logic: If the loop completes without identifying a difference, the strings are identical.
   */
  if (!difference_found) {
    cout << "0";
  }
  
  return 0;
}
