/**
 * @file 118A.CPP
 * @brief Solution for Codeforces Problem 118A, "String Task".
 * @details This program processes a string by applying a set of rules:
 *          1. Delete all vowels (A, E, I, O, U, Y), case-insensitively.
 *          2. Insert a '.' character before each consonant.
 *          3. Convert all uppercase consonants to their lowercase equivalents.
 *
 * @section
 *
 * Algorithm:
 * The program iterates through the input string character by character and builds a
 * new result string. For each character, it checks if it is a vowel.
 * - If it's a vowel, the character is ignored.
 * - If it's a consonant, a '.' is first appended to the result string. Then, the
 *   consonant itself is converted to lowercase and also appended.
 *
 * The string concatenation (`newLine = newLine + ...`) inside the loop can be
 * inefficient, potentially leading to a time complexity of O(L^2) due to
 * repeated reallocations.
 *
 * - Time Complexity: O(L^2) in the worst case, where L is the length of the string.
 * - Space Complexity: O(L) for the input and result strings.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string line, newLine = "";

  cin >> line;

  /**
   * Block Logic: Iterates through each character of the input string 'line'.
   */
  for (int i = 0; i < line.length(); i++) {
    /**
     * Block Logic: This block handles the three cases for each character:
     * 1. Vowel: The character is skipped.
     * 2. Uppercase Consonant: A '.' is prepended, and the char is lowercased.
     * 3. Lowercase Consonant: A '.' is prepended.
     */
    if (line[i] == 'a' || line[i] == 'A' || line[i] == 'e' || line[i] == 'E' ||
        line[i] == 'i' || line[i] == 'I' || line[i] == 'o' || line[i] == 'O' ||
        line[i] == 'u' || line[i] == 'U' || line[i] == 'y' || line[i] == 'Y')
      continue;
    else if (isupper(line[i]))
      newLine = newLine + '.' + char(tolower(line[i]));
    else
      newLine = newLine + '.' + line[i];
  }

  cout << newLine << endl;

  return 0;
}
