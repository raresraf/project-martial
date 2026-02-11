/**
 * @file 118A.CPP
 * @brief Processes a string by removing vowels and prepending '.' to consonants.
 * @details This program is a solution for the "String Task" on Codeforces. It reads a string,
 * converts it to lowercase, removes all vowels ('a', 'o', 'y', 'e', 'u', 'i'), and inserts
 * a '.' character before each consonant. The implementation is well-structured, using helper
 * functions for character conversion and vowel identification.
 *
 * @see https://codeforces.com/problemset/problem/118/A
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Create a new, empty string to build the result.
 * 3. Iterate through each character of the input string.
 * 4. For each character, use a helper function `isVowel` to check if it's a vowel.
 * 5. If it is a vowel, ignore it and continue to the next character.
 * 6. If it is a consonant, convert it to lowercase, prepend a '.' to it, and append
 *    the combination to the result string.
 * 7. After the loop, print the final resulting string.
 *
 * Time Complexity: O(N), where N is the length of the input string, as it performs a single pass.
 * Space Complexity: O(N), as a new string is created which can be up to twice the length of the input.
 */
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Converts an uppercase character to lowercase.
 * @param ch The character to convert.
 * @return The lowercase version of the character, or the character itself if it's not uppercase.
 * @note This implementation assumes ASCII encoding. `ch - 'a' < 0` checks if `ch` is before 'a', which is true for uppercase letters.
 */
char convertToLowerCase(char ch) {
  if (ch >= 'A' && ch <= 'Z') // A more standard way to check for uppercase
    return ch + 32;
  return ch;
}

/**
 * @brief Checks if a character is a vowel.
 * @param ch The character to check.
 * @return True if the character is a vowel ('a', 'o', 'y', 'e', 'u', 'i'), false otherwise.
 * @note The check is case-insensitive.
 */
bool isVowel(char ch) {
  ch = convertToLowerCase(ch);
  if (ch == 'a' || ch == 'o' || ch == 'y' || ch == 'e' || ch == 'u' ||
      ch == 'i')
    return true;
  return false;
}

int main() {
  // s: the original input string.
  // updated_string: the result string being built.
  string s, updated_string = "";
  cin >> s;

  /**
   * @brief Main loop to iterate through the input string and build the output string.
   *
   * Pre-condition: 's' holds the user's input string.
   * Invariant: `updated_string` contains the correctly processed version of the characters seen so far.
   */
  for (char ch : s) {
    // If the character is a vowel, skip it.
    if (isVowel(ch))
      continue;
    
    // If it's a consonant, append '.' and the lowercase character to the result.
    updated_string += "." + string(1, convertToLowerCase(ch));
  }

  // Print the final processed string.
  cout << updated_string << endl;
  return 0;
}
