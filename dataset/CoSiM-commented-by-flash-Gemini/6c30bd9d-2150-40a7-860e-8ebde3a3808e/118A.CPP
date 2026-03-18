/**
 * @file 118A.CPP
 * @brief Solution to a competitive programming problem: "String Task".
 * Processes an input string, converting uppercase letters to lowercase,
 * removing all vowels, and inserting a "." before each consonant.
 * Algorithm: Character-by-character processing with conditional logic.
 * Time Complexity: O(L) where L is the length of the input string.
 * Space Complexity: O(1) for output, excluding input string storage.
 */
#include <iostream>

#define EMPTY_STRING ""

/**
 * @brief Checks if a given character is a vowel (case-insensitive).
 * @param c The character to check.
 * @return True if the character is a vowel (a, e, i, o, u, y), false otherwise.
 */
bool isVowel(char c) {
  return ((c == 'a') || (c == 'A') || (c == 'e') || (c == 'E') || (c == 'i') ||
          (c == 'I') || (c == 'o') || (c == 'O') || (c == 'u') || (c == 'U') ||
          (c == 'y') || (c == 'Y'));
}

/**
 * @brief Processes a character according to the problem rules.
 * If the character is a vowel, returns an empty string.
 * If it's a consonant, returns a string with a "." prefix and the lowercase version of the character.
 * @param c The character to process.
 * @return A string containing the processed character or an empty string.
 */
std::string replace(char c) {
  std::string ret = EMPTY_STRING;
  /**
   * Block Logic: If the character is a vowel, it should be removed.
   * Pre-condition: 'c' is a valid character.
   * Invariant: If 'c' is a vowel, 'ret' remains empty.
   */
  if (isVowel(c))
    return ret;
  
  ret += ".";
  /**
   * Block Logic: Converts uppercase consonants to lowercase.
   * Pre-condition: 'c' is a consonant.
   * Invariant: 'c' is appended to 'ret' as a lowercase character.
   */
  if ((c >= 'a') && (c <= 'z'))
    ret += c;
  else if ((c >= 'A') && (c <= 'Z'))
    ret += c - 'A' + 'a';
  return ret;
}

/**
 * @brief Main function to solve the String Task problem.
 * Reads an input string, processes each character, and prints the modified string.
 */
int main() {
  std::string s;
  std::cin >> s;
  /**
   * Block Logic: Iterates through each character of the input string.
   * Pre-condition: 's' contains the input string.
   * Invariant: Each character is processed by the 'replace' function and its result is printed.
   */
  for (unsigned int i = 0; i < s.size(); ++i) {
    std::cout << replace(s[i]);
  }
  std::cout << '\n';
}

