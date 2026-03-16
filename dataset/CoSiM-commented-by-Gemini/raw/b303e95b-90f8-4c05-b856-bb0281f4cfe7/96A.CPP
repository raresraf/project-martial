/**
 * @file 96A.CPP
 * @brief Solution to Codeforces problem 96A - "Football".
 * @details The program determines if a "dangerous" situation has occurred, which is defined
 *          as having at least seven consecutive players of the same team ('0' or '1').
 *
 * Algorithm:
 * 1. Read the input string representing the players' positions.
 * 2. Iterate through the string to find a sequence of 7 identical, consecutive characters.
 *    - The implementation uses a complex and inefficient nested loop structure to check
 *      for a substring of length 7 where all characters are the same.
 *    - For each character `s[i]`, it attempts to verify that the next 6 characters
 *      (`s[i+1]` through `s[i+6]`) are identical to `s[i]`.
 * 3. If such a sequence is found, print "YES" and terminate.
 * 4. If the loop completes without finding any such sequence, print "NO".
 *
 * Note: The implementation has a potential for out-of-bounds memory access because the
 * loop condition `i < s.size()` does not prevent `s[i+6]` from reading past the
 * end of the string. A safer condition would be `i <= s.size() - 7`.
 *
 * Time Complexity: O(N) in the best case (finds a match early) and worst case, as it makes a single pass.
 *                  However, the nested structure adds unnecessary complexity.
 * Space Complexity: O(N) for storing the input string.
 */
#include <iostream>
#include <string>
using namespace std;
int main() {
  string s;
  int i, j;
  cin >> s;
  /**
   * @brief Main loop to check for a dangerous situation.
   * It iterates through the string, and for each position `i`, it checks if the
   * following 6 characters are the same as the character at `i`.
   */
  for (i = 0; i < s.size(); i++) {
    // Pre-condition check to see if a 7-character sequence starting at `i` could possibly match.
    // This also introduces a risk of out-of-bounds access if `i` is near the end of the string.
    if (s[i] == s[i + 6]) {
      /**
       * @brief Inner loop to verify the intermediate characters.
       * It checks if characters from `s[i+1]` to `s[i+5]` are also the same as `s[i]`.
       */
      for (j = 1; j < 6; j++) {
        // This block executes on the last iteration of the inner loop (j=5).
        if (j == 5 && s[i + j] == s[i]) {
          // If all characters from s[i] to s[i+6] have matched, a dangerous situation is found.
          cout << "YES";
          return 0;
        } else {
          // If an intermediate character matches, continue checking.
          if (s[i + j] == s[i])
            continue;
          else
            // If any character breaks the sequence, stop checking from this `i`.
            break;
        }
      }
    }
  }
  // If the loop completes, no sequence of 7 was found.
  cout << "NO";
  return 0;
}
