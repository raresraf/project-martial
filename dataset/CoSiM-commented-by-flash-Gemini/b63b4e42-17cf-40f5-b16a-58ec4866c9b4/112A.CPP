/**
 * @file 112A.CPP
 * @brief Case-insensitive lexicographical comparison of two strings.
 * 
 * Logic: Compares two input strings character-by-character while ignoring 
 * case differences. Normalizes both characters to lowercase before comparison. 
 * Returns -1, 0, or 1 based on the relative dictionary order.
 * 
 * Algorithm: Single-pass scan with character-level normalization.
 * Time Complexity: $O(N)$ where N is the length of the strings.
 * Space Complexity: $O(N)$ for string storage.
 */

#include <iostream>
using namespace std;

int main() {
  string first;
  // Functional Utility: Reads the first full string including any whitespace.
  if (!getline(cin, first)) return 0;
  
  string second;
  // Functional Utility: Reads the second full string.
  if (!getline(cin, second)) return 0;

  // State: Result of the lexicographical comparison (default to identity/0).
  int comparison = 0;

  /**
   * Block Logic: Parallel traversal of strings for order evaluation.
   * Invariant: Processes characters until a mismatch is found or strings are exhausted.
   */
  for (int k = 0; k < (int)first.size(); ++k) {
    /**
     * Logic: Case-normalized comparison.
     * Uses tolower() to map uppercase ASCII to lowercase for consistent evaluation.
     */
    if (tolower(first[k]) < tolower(second[k])) {
      // Case 1: 'first' precedes 'second' in dictionary order.
      comparison = -1;
      break;
    }
    if (tolower(first[k]) > tolower(second[k])) {
      // Case 2: 'first' follows 'second' in dictionary order.
      comparison = 1;
      break;
    }
  }

  // Output the comparison status (-1, 0, or 1).
  printf("%d\n", comparison);
  
  return 0;
}
