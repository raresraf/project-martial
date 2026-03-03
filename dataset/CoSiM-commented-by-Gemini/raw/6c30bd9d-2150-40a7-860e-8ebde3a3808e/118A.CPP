
#include <iostream>

#define EMPTY_STRING ""

/**
 * @file 118A.CPP
 * @brief Solution to Codeforces problem 118A, "String Task".
 * @details The program processes a string by applying a set of rules:
 *          1. Delete all vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 *          2. Insert a '.' character before each consonant.
 *          3. Convert all uppercase consonants to their lowercase equivalents.
 *
 * Algorithm:
 * The program iterates through the input string character by character. For each
 * character, it first determines if it is a vowel. If it is, the character is
 * discarded. If it is a consonant, it is converted to lowercase, and a '.'
 * character is prepended to it. The resulting transformed characters (or empty
 * strings for vowels) are concatenated to form the final output string.
 *
 * Time Complexity: O(N), where N is the length of the input string, as each
 * character is processed exactly once.
 * Space Complexity: O(N) in the worst case (a string of all consonants), as a
 * new string is effectively built.
 */

/**
 * @brief Checks if a character is a vowel (case-insensitive).
 * The set of vowels includes 'A', 'O', 'Y', 'E', 'U', 'I'.
 * @param c The character to check.
 * @return True if the character is a vowel, false otherwise.
 */
bool isVowel(char c) {
  return ((c == 'a') || (c == 'A') || (c == 'e') || (c == 'E') || (c == 'i') ||
          (c == 'I') || (c == 'o') || (c == 'O') || (c == 'u') || (c == 'U') ||
          (c == 'y') || (c == 'Y'));
}

/**
 * @brief Processes a single character according to the problem rules.
 * @param c The character to process.
 * @return A string representing the transformed character. An empty string for
 *         vowels, or a ".c" formatted string for consonants.
 */
std::string replace(char c) {
  std::string ret = EMPTY_STRING;
  // Rule 1: Delete vowels.
  if (isVowel(c))
    return ret;

  // Rule 2: Insert '.' before consonants.
  ret += ".";

  // Rule 3: Convert uppercase consonants to lowercase.
  if ((c >= 'a') && (c <= 'z'))
    ret += c;
  else if ((c >= 'A') && (c <= 'Z'))
    ret += c - 'A' + 'a'; // Convert to lowercase.
  return ret;
}

int main() {
  std::string s;
  std::cin >> s;

  /**
   * @brief Main processing loop.
   * Iterates through the input string and prints the transformed version of
   * each character.
   */
  for (unsigned int i = 0; i < s.size(); ++i) {
    std::cout << replace(s[i]);
  }
  std::cout << '\n';
}
