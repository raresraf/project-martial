/**
 * @file 118A.CPP
 * @brief Solution for the "String Task" problem.
 *
 * @details
 * This program processes a string according to a specific set of rules:
 * 1. It deletes all vowels.
 * 2. It inserts a "." character before each consonant.
 * 3. It replaces all uppercase consonants with their lowercase counterparts.
 *
 * The implementation uses a two-pass approach. The first pass processes the
 * string in-place, marking vowels with a sentinel character ('-') and converting
 * uppercase consonants to lowercase. The second pass constructs the final output
 * string from this intermediate representation.
 *
 * @section
 * Algorithm: Two-pass in-place normalization and output generation.
 * Time Complexity: O(N), where N is the length of the string, due to the two separate loops.
 * Space Complexity: O(N) to store the input C-style string.
 */
#include <cstring>
#include <iostream>

using namespace std;
int main() {
  int i;
  char str[100]; // C-style character array to hold the input string.
  cin >> str;

  /**
   * @brief First pass: Normalize the string in-place.
   *
   * Logic: This loop iterates through the string. Vowels are replaced by a
   * sentinel character '-'. Uppercase consonants are converted to lowercase.
   */
  // string processing
  for (i = 0; i < strlen(str); i++) {
    // Pre-condition: Check if the character is any vowel (case-sensitive).
    if ((str[i] == 'A') || (str[i] == 'O') || (str[i] == 'Y') ||
        (str[i] == 'E') || (str[i] == 'U') || (str[i] == 'I') ||
        (str[i] == 'a') || (str[i] == 'o') || (str[i] == 'y') ||
        (str[i] == 'e') || (str[i] == 'u') || (str[i] == 'i'))
      // Invariant: Mark vowels with a hyphen for later processing.
      str[i] = '-';
    // Else, check if the character is an uppercase letter using its ASCII value.
    else if ((64 < (int)str[i]) && ((int)str[i] < 91))
      // Convert the uppercase consonant to lowercase by adding 32 to its ASCII value.
      str[i] = (char)((int)str[i] + 32);
  }

  /**
   * @brief Second pass: Print the final processed string.
   *
   * Logic: This loop iterates through the modified string. It skips the sentinel
   * hyphens and prints a '.' followed by the remaining (consonant) characters.
   */
  // string printing
  for (i = 0; i < strlen(str); i++) {
    if (str[i] != '-')
      cout << "." << str[i];
  }
  cout << endl;
  return 0;
}
