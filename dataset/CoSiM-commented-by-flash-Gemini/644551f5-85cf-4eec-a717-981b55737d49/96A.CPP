/**
 * @file 96A.CPP
 * @brief This program solves the "Football" problem (Codeforces 96A).
 *   It determines if a given string of '0's and '1's represents a "dangerous" situation.
 *   A situation is considered dangerous if there are 7 or more consecutive '0's
 *   OR 7 or more consecutive '1's in the string.
 *   Problem description: https://codeforces.com/problemset/problem/96/A
 * * Algorithm: String traversal with consecutive count tracking.
 *   It iterates through the string, maintaining separate counters for consecutive '0's and '1's.
 *   If either counter reaches 7, the situation is deemed dangerous. Counters are reset
 *   when the character sequence breaks.
 * Time Complexity: O(L) where L is the length of the input string, as it involves
 *   a single (or effectively two) pass(es) through the string.
 * Space Complexity: O(L) for storing the input string.
 */
#include <iostream> // Required for standard input/output operations.
#include <string>   // Required for std::string.

using namespace std;

/**
 * @brief Main function to check if a football situation is "dangerous".
 * Functional Utility: Reads a string of '0's and '1's, checks for 7 or more
 *                     consecutive identical characters, and prints "YES" or "NO".
 */
int main() {
  int len, i;      // `len` for string length, `i` as loop counter.
  int n = 0;       // Counter for consecutive '1's.
  int m = 0;       // Counter for consecutive '0's.
  int a = 0;       // Total count of '0's in the string.
  int b = 0;       // Total count of '1's in the string.
  string str;      // Stores the input string.

  // Block Logic: Reads the input string.
  cin >> str;
  len = str.length(); // Gets the length of the input string.

  /**
   * Block Logic: First pass to count the total occurrences of '0's and '1's.
   * This information is used for an initial check, although the condition
   * `if (a > 0 && b > 0)` has a potential flaw (see below).
   */
  for (i = 0; i < len; i++) {
    if (str[i] == '0')
      a++;
    if (str[i] == '1')
      b++;
  }

  /**
   * Conditional Logic: The condition `if (a > 0 && b > 0)` checks if the string contains
   * both '0's and '1's. This is a potential flaw in the problem interpretation or implementation,
   * as a string like "0000000" (which is dangerous) would not satisfy this condition,
   * leading to incorrect output ("NO" instead of "YES").
   * The problem statement typically implies checking for consecutive sequences regardless of mixed content.
   */
  if (a > 0 && b > 0) { // This condition implicitly handles strings composed solely of one character.
    /**
     * Block Logic: Second pass through the string to track consecutive occurrences of '0's and '1's.
     * Invariant: `n` tracks consecutive '1's, `m` tracks consecutive '0's. If either reaches 7, "YES" is printed.
     */
    for (i = 0; i < len; i++) {
      // Block Logic: Handles counting consecutive '1's.
      if (str[i] == '1') {
        n++; // Increment consecutive '1's counter.
        // Conditional Logic: If 7 or more consecutive '1's are found, print "YES" and terminate.
        if (n == 7) {
          cout << "YES";
          return 0; // Early exit.
        }
        // Conditional Logic: If the next character is '0', the sequence of '1's is broken, reset counter.
        // Inline: `i + 1` requires checking bounds (`i + 1 < len`) to prevent out-of-bounds access.
        // This is implicitly handled if the problem constraints guarantee string length > 0.
        // A more robust check would be `if (i + 1 < len && str[i + 1] == '0')`.
        if (str[i + 1] == '0') {
          n = 0;
        }
      }

      // Block Logic: Handles counting consecutive '0's.
      if (str[i] == '0') {
        m++; // Increment consecutive '0's counter.
        // Conditional Logic: If 7 or more consecutive '0's are found, print "YES" and terminate.
        if (m == 7) {
          cout << "YES";
          return 0; // Early exit.
        }
        // Conditional Logic: If the next character is '1', the sequence of '0's is broken, reset counter.
        // Inline: Similar to above, `i + 1 < len` check is implicitly assumed or required.
        if (str[i + 1] == '1') {
          m = 0;
        }
      }
    }
    // If the loop completes without finding 7 consecutive identical characters, print "NO".
    cout << "NO";
  }
  return 0;
}
