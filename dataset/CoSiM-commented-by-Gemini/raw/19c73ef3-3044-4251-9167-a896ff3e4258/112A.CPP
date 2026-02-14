/**
 * @file 112A.CPP
 * @brief Solution for a problem requiring case-insensitive lexicographical string comparison.
 * @details This program compares two input strings lexicographically, ignoring case. It does this
 * by comparing the alphabetical order of characters one by one without modifying the original strings.
 * It outputs -1 if the first string is smaller, 1 if it's larger, and 0 if they are equal.
 *
 * @section algorithm Algorithm
 * 1. Read two strings, `p` and `s`.
 * 2. Iterate through the strings from the first character to the last.
 * 3. In each iteration, for the characters at the current position `i` in both strings:
 *    a. Normalize the character to a zero-based alphabetical index (A/a -> 0, B/b -> 1, etc.).
 *       This is done by checking if the character's ASCII value is in the uppercase range (65-90)
 *       or lowercase range (97-122) and subtracting the base value (65 or 97).
 *    b. Compare the two normalized values.
 *    c. If the value from the first string is less than the second, the first string is
 *       lexicographically smaller. Print "-1" and terminate the program.
 *    d. If the value from the first string is greater, the first string is larger.
 *       Print "1" and terminate.
 * 4. If the loop completes without finding any difference, the strings are considered equal.
 *    Print "0".
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(L), where L is the length of the strings. In the worst case (equal strings),
 *   it iterates through the entire length. In the best case, it stops at the first differing character.
 * - Space Complexity: O(L) to store the two input strings.
 */
#include <bits/stdc++.h>
#include <cctype> // For std::tolower

using namespace std;

/**
 * @brief Main function to perform case-insensitive string comparison.
 * @return 0 on successful execution.
 */
int main() {
  string p, s;
  cin >> p >> s;

  for (int i = 0; i < p.length(); i++) {
    // Convert characters to a common case (lowercase) for comparison.
    char char1 = tolower(p[i]);
    char char2 = tolower(s[i]);
    
    if (char1 < char2) {
      cout << "-1";
      return 0; // Exit as soon as a difference is found.
    }
    if (char1 > char2) {
      cout << "1";
      return 0; // Exit as soon as a difference is found.
    }
  }

  // If the loop completes, the strings are equal.
  cout << "0";
  return 0;
}
