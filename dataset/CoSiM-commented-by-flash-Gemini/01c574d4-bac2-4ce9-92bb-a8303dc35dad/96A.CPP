/**
 * @file 96A.CPP
 * @brief Solution to Codeforces Problem 96A: Football.
 * This program determines if a football team is "dangerous" based on a sequence of ball possessions.
 * A team is dangerous if any player has the ball for 7 or more consecutive minutes.
 * This translates to finding if there are 7 consecutive identical characters in the input string.
 * Algorithm: Iterative scan with a fixed-window check.
 * Time Complexity: O(N) where N is the length of the input string.
 * Space Complexity: O(N) for storing the input string.
 */

#include <bits/stdc++.h>

using namespace std;

string s;

main() {
  // Functional Utility: Reads the input string representing ball possessions.
  cin >> s;

  // Block Logic: Pre-check to ensure the string is long enough to contain a 7-minute dangerous sequence.
  // Precondition: s.size() >= 7 is required to even attempt a 7-consecutive-character check.
  // Invariant: If s.size() < 7, no dangerous sequence is possible, and the program will directly output "NO".
  if (s.size() >= 7) {

    // Block Logic: Iterates through the string to find a 7-minute dangerous sequence.
    // The loop runs up to `s.size() - 6` to ensure that there are always 7 characters remaining to check (i.e., s[i] to s[i+6]).
    // Invariant: At each iteration, `i` is the potential starting index of a 7-character sequence.
    for (int i = 0; i < s.size() - 6; i++) {
      //	cout<<"1"; // Debugging comment in original code.
      
      // Variable to count differing characters within the 7-character window.
      int k = 0;
      // 'r' stores the character at the start of the current 7-character window.
      char r = s[i];

      // Block Logic: Checks the 7 characters starting from index `i` for uniformity.
      // Invariant: All characters in the window [i, i+6] are compared against `s[i]`.
      for (int j = i; j < i + 7; j++) {
        // Conditional Logic: Increments 'k' if a character in the window differs from the starting character.
        if (r != s[j]) {
          k++;
        }
      }

      // Conditional Logic: If 'k' is 0, it means all 7 characters were identical.
      // This indicates a "dangerous" team, so output "YES" and terminate.
      if (k == 0) {
        cout << "YES";
        return 0;
      }
    }
  }

  // Block Logic: If the loop completes without finding a dangerous sequence, output "NO".
  cout << "NO";
  return 0;
}
