/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from competitive programming.
 *
 * @details
 * This program processes a string by removing vowels, prepending consonants with
 * a '.', and converting all consonants to lowercase.
 *
 * Algorithm:
 * 1. Read the entire input string.
 * 2. Iterate through each character of the string.
 * 3. For each character, convert it to lowercase to simplify vowel checking.
 * 4. If the character is a vowel ('a', 'e', 'i', 'o', 'u', 'y'), ignore it and continue.
 * 5. If the character is a consonant, print a '.' followed by the lowercase character.
 *
 * This implementation reads the whole string into memory but generates the output
 * in a stream-like fashion without building a separate output string.
 *
 * Time Complexity: O(N), where N is the length of the string.
 * Space Complexity: O(N) to store the input string.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  int i;
  string s;
  cin >> s;
  /**
   * @brief Iterates through each character of the input string.
   */
  for (i = 0; i < s.length(); i++) {
    // Convert the current character to lowercase for case-insensitive comparison.
    char x = tolower(s[i]);
    // Check if the character is a vowel.
    if (x == 'a' || x == 'e' || x == 'i' || x == 'o' || x == 'u' || x == 'y') {
      // If it's a vowel, skip to the next character.
      continue;
    } else
      // If it's a consonant, print a '.' followed by the lowercase character.
      cout << "." << x;
  }
  return 0;
}