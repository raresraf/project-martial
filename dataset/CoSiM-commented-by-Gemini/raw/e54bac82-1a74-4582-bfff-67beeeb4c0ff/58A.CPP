/**
 * @file 58A.CPP
 * @brief Subsequence check for the word "hello".
 * @details This program determines if the string "hello" is a subsequence of a given input string.
 * It reads a string from standard input and checks if the characters 'h', 'e', 'l', 'l', 'o' appear in that order,
 * but not necessarily contiguously.
 *
 * @algorithm Subsequence String Matching
 * @complexity
 * - Time: O(N), where N is the length of the input string. The program iterates through the input string once.
 * - Space: O(1), constant extra space is used, ignoring the space for the input string itself.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  // The target subsequence to find.
  string t = "hello";
  // The input string from the user.
  string s;
  cin >> s;

  // j is an index for the target string 't'.
  int j = 0;
  /**
   * @block Pre-condition: 's' is the input string, 't' is "hello".
   * This loop iterates through each character of the input string 's'.
   * @invariant At any point, 'j' holds the number of characters from "hello" that have been matched so far.
   */
  for (int i = 0; s[i] != '\0'; i++) {
    /**
     * @block Logic: Checks if the current character in the input string 's'
     * matches the current character in the target subsequence 't'.
     */
    if (s[i] == t[j]) {
      // Mark the character in 't' as found and advance the subsequence index.
      // This is a non-standard way to track progress; it modifies the target string.
      t[j++] = 0;
    }
  }

  /**
   * @block Logic: Checks if the entire subsequence has been found.
   * If the last character of 't' has been marked as 0, it implies all preceding
   * characters were also matched in the correct order.
   */
  if (t[t.length() - 1] == 0)
    cout << "YES
";
  else
    cout << "NO
";

  return 0;
}
