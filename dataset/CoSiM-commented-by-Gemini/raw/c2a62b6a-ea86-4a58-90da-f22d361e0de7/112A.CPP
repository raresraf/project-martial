/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem (Codeforces 112A).
 * @details This program compares two strings lexicographically, ignoring the case of the letters.
 *
 * Problem Description:
 * Little Petya wants to compare two strings lexicographically. The letters' case
 * does not matter; an uppercase letter is considered equivalent to the corresponding
 * lowercase letter. The program should output -1 if the first string is less than
 * the second, 1 if the second is less than the first, and 0 if they are equal.
 *
 * Algorithm: Case-Insensitive Lexicographical Comparison
 * 1. The program first reads two strings of the same length.
 * 2. It converts both strings entirely to lowercase to ensure the comparison is case-insensitive.
 * 3. It then iterates through the normalized strings character by character.
 * 4. At the first position where the characters differ, it determines which string is lexicographically
 *    greater and prints the corresponding value (1 or -1).
 * 5. If the loop completes without finding any differences, the strings are considered equal, and 0 is printed.
 *
 * Time Complexity: O(L), where L is the length of the strings, due to two passes over the strings.
 * Space Complexity: O(L), to store the two input strings.
 */

#include <iostream>
#include <string>
using namespace std;

int main() {

  string str1, str2;
  int i;
  cin >> str1;
  cin >> str2;
  int len = str1.length();
  /**
   * Block Logic: Normalize both strings to lowercase.
   * This step is crucial for the case-insensitive comparison required by the problem.
   */
  for (int i = 0; i < len; i++) {
    str1[i] = tolower(str1[i]);
    str2[i] = tolower(str2[i]);
  }
  /**
   * Block Logic: Perform character-by-character comparison.
   * Pre-condition: Both strings are of the same length and are in lowercase.
   * The loop will terminate at the first differing character.
   */
  for (i = 0; i < str1.length(); i++) {
    // Compare ASCII values of the characters.
    if (str1[i] - str2[i] > 0) {
      cout << 1 << endl;
      break;
    } else if (str1[i] - str2[i] < 0) {
      cout << -1 << endl;
      break;
    } else
      // Characters are the same, continue to the next pair.
      continue;
  }

  /**
   * Block Logic: Handle the case of equal strings.
   * If the loop finished without breaking, it means all characters were identical.
   * The loop counter 'i' will equal the string length in this scenario.
   */
  if (i == str1.length())
    cout << 0 << endl;

  return 0;
}
