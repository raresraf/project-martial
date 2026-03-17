/**
 * @file 112A.CPP
 * @brief Solution to the "Petya and Strings" problem.
 *
 * @details This program compares two input strings lexicographically, ignoring case.
 * It reads two strings, converts them to the same case (lowercase), and then determines
 * their lexicographical order.
 *
 * @section
 * Algorithm:
 * 1. Read two strings, s and t.
 * 2. Iterate through the strings from the first character to the last.
 * 3. In each iteration, convert the current characters from both strings to lowercase.
 * 4. Compare the lowercase characters.
 *    - If the character from the first string is greater, the first string is lexicographically larger. Print 1 and exit.
 *    - If the character from the first string is smaller, the first string is lexicographically smaller. Print -1 and exit.
 * 5. If the loop completes without any difference found, the strings are equal. Print 0.
 *
 * Time Complexity: O(L), where L is the length of the strings, as it's a single pass.
 * Space Complexity: O(L), to store the two input strings.
 */
#include <iostream>

using namespace std;

int main() {
  string s, t;
  cin >> s >> t;
  /**
   * @brief This loop iterates through both strings simultaneously to find the first
   * lexicographical difference, ignoring case.
   */
  for (int i = 0; i < s.size(); ++i) {
    // Block Logic: Convert characters to lowercase for case-insensitive comparison.
    if (s[i] >= 'A' && s[i] <= 'Z')
      s[i] = s[i] + 'a' - 'A';
    if (t[i] >= 'A' && t[i] <= 'Z')
      t[i] = t[i] + 'a' - 'A';
    
    /**
     * @brief Compares the characters at the current position.
     * Logic: If a difference is found, print the corresponding result (-1 or 1) and terminate.
     * The program only continues if the characters are identical.
     */
    if (s[i] > t[i]) {
      cout << 1;
      return 0;
    } else if (s[i] < t[i]) {
      cout << -1;
      return 0;
    }
  }
  // If the loop finishes, the strings are lexicographically equal.
  cout << 0;
  return 0;
}