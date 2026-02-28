/**
 * @file 118A.CPP
 * @brief Solution for the "String Task" problem from Codeforces.
 * @details The problem requires processing a string with three rules:
 * 1. Delete all vowels (A, O, Y, E, U, I).
 * 2. Insert a character "." before each consonant.
 * 3. Replace all uppercase consonants with their lowercase equivalent.
 * This solution builds a new result string in a single pass.
 *
 * @algorithm Single-Pass String Building
 * 1. Read the input string.
 * 2. Create an empty result string.
 * 3. Iterate through each character of the input string.
 * 4. For each character, check if it is a vowel (case-insensitively).
 * 5. If it is a vowel, do nothing (effectively deleting it).
 * 6. If it is a consonant, append a '.' to the result string, then append the
 *    lowercase version of the consonant to the result string.
 * 7. After the loop, print the final result string.
 *
 * @complexity
 * - Time: O(N), where N is the length of the input string, as it's processed in a single pass.
 * - Space: O(N), to store both the input and the result strings.
 */
#include <string>
#include <iostream>
#include <cctype> // For towlower

using namespace std;

int main(int argc, const char *argv[]) {
  // `str` for the input string, `res` to build the result string.
  string str, res;
  cin >> str;

  /**
   * @brief Iterate through the input string to process each character.
   */
  // Invariant: `res` contains the correctly formatted output for the prefix of
  // `str` that has been processed so far.
  for (int i = 0; i < str.length(); i++) {
    // Pre-condition: Check if the character is a vowel (case-insensitive).
    if (str[i] == 'a' || str[i] == 'e' || str[i] == 'i' || str[i] == 'o' ||
        str[i] == 'u' || str[i] == 'y' || str[i] == 'A' || str[i] == 'E' ||
        str[i] == 'O' || str[i] == 'I' || str[i] == 'U' || str[i] == 'Y')
      // If it's a vowel, skip to the next character.
      continue;
    else {
      // If it's a consonant, append a '.' to the result.
      res += '.';
      // Then, append the lowercase version of the consonant to the result.
      res += towlower(str[i]);
    }
  }
  
  // Print the final constructed string.
  cout << res;

  return 0; // Indicate successful execution.
}
