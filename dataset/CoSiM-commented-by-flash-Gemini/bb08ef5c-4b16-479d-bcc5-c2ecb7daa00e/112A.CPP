/**
 * @file 112A.CPP
 * @brief Case-insensitive lexicographical comparison of two input strings.
 * 
 * Logic: Compares two strings character-by-character while ignoring case differences. 
 * Normalizes uppercase ASCII characters to lowercase on-the-fly and determines 
 * the relative dictionary order.
 * 
 * Algorithm: Single-pass scan with character normalization and early termination.
 * Time Complexity: $O(N)$ where N is the length of the strings.
 * Space Complexity: $O(N)$ for the input string buffers.
 */

#include <iostream>

using namespace std;

int main() {
  string s, t;
  // Functional Utility: Reads two character sequences from standard input.
  if (!(cin >> s >> t)) return 0;

  /**
   * Block Logic: Parallel traversal of strings for case-normalized comparison.
   * Invariant: Evaluates character pairs until a mismatch is found or strings are exhausted.
   */
  for (int i = 0; i < (int)s.size(); ++i) {
    /**
     * Logic: Normalization.
     * Maps uppercase ASCII characters ['A'-'Z'] to lowercase ['a'-'z'] by applying 
     * the standard fixed offset.
     */
    if (s[i] >= 'A' && s[i] <= 'Z')
      s[i] = (char)(s[i] + ('a' - 'A'));
    if (t[i] >= 'A' && t[i] <= 'Z')
      t[i] = (char)(t[i] + ('a' - 'A'));

    /**
     * Functional Utility: Lexicographical order evaluation.
     * Logic: Returns 1 if 's' is greater, -1 if 't' is greater, and terminates.
     */
    if (s[i] > t[i]) {
      cout << 1;
      return 0;
    } else if (s[i] < t[i]) {
      cout << -1;
      return 0;
    }
  }

  /**
   * Fallback: Identity condition.
   * Output 0 if no differences were identified during the scan.
   */
  cout << 0;
  
  return 0;
}
