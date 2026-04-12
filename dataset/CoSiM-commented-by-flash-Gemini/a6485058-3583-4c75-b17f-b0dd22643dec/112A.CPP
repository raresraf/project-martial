/**
 * @file 112A.CPP
 * @brief Case-insensitive lexicographical string comparison utility.
 * 
 * Logic: Compares two input strings by normalizing them to uppercase and evaluating 
 * their relative dictionary order character by character.
 * 
 * Algorithm: Single-pass scan with ASCII-based state normalization.
 * Time Complexity: $O(N)$ where N is the length of the strings.
 * Space Complexity: $O(N)$ for the string buffers.
 */

#include <cstring>
#include <iostream>

using namespace std;

int main() {
  string a, b;
  // Functional Utility: Reads two strings from standard input.
  cin >> a >> b;

  // State: Comparison result (-1, 0, or 1). Initialized to an arbitrary non-result value.
  int k = 2;

  /**
   * Block Logic: Parallel traversal of input strings for character comparison.
   * Invariant: Normalizes characters to uppercase before comparison to ensure 
   * case-insensitivity.
   */
  for (int i = 0; i < a.length(); i++) {
    /**
     * Logic: Uppercase normalization.
     * Maps lowercase ASCII [a-z] (97-122) to uppercase [A-Z] by subtracting 32.
     */
    if (a[i] >= 97) {
      a[i] = a[i] - 32;
    }

    if (b[i] >= 97) {
      b[i] = b[i] - 32;
    }

    /**
     * Functional Utility: Lexicographical order determination.
     * Logic: If a difference is found, set result 'k' and terminate the scan.
     */
    if (a[i] < b[i]) {
      k = -1;
      break;
    } else if (a[i] > b[i]) {
      k = 1;
      break;
    } else {
      // Logic: Strings remain identical up to index 'i'.
      k = 0;
    }
  }

  // Output the comparison status (-1: a < b, 1: a > b, 0: a == b).
  cout << k << endl;

  return 0;
}
