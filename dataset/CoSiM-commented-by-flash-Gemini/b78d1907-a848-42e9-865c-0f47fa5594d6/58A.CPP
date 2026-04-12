/**
 * @file 58A.CPP
 * @brief Subsequence detection algorithm using nested-loop structural matching.
 * 
 * Logic: Determines if the word "hello" exists as a subsequence within the input 
 * string 's'. Employs a sequence of nested scans to ensure that the characters 
 * 'h', 'e', 'l', 'l', 'o' appear in the strictly correct relative order.
 * 
 * Algorithm: Exhaustive nested searching for pattern components.
 * Time Complexity: $O(N^5)$ in the worst case (though practically limited by early exit and pattern size).
 * Space Complexity: $O(N)$ for string storage.
 */

#include <iostream>
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  string s;
  // Functional Utility: Reads the input character sequence.
  if (!(cin >> s)) return 0;
  n = (int)s.length();

  /**
   * Block Logic: Deeply nested structural matching for the target subsequence.
   * Invariant: Each nested loop level searches for the next character in the 
   * sequence "hello" starting from the index immediately following the previous match.
   */
  for (int i = 0; i < n; i++) {
    // Phase 1: Search for the initial anchor 'h'.
    if (s[i] == 'h') {
      for (int j = i + 1; j < n; j++) {
        // Phase 2: Search for 'e' following 'h'.
        if (s[j] == 'e') {
          for (int k = j + 1; k < n; k++) {
            // Phase 3: Search for the first 'l' following 'e'.
            if (s[k] == 'l') {
              for (int l = k + 1; l < n; l++) {
                // Phase 4: Search for the second 'l' following the first 'l'.
                if (s[l] == 'l') {
                  for (int m = l + 1; m < n; m++) {
                    // Phase 5: Search for the terminating 'o' following 'l'.
                    if (s[m] == 'o') {
                      // Functional Utility: Final state reached.
                      cout << "YES" << endl;
                      return 0;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // Fallback: The target sequence was not identified.
  cout << "NO" << endl;
  
  return 0;
}
