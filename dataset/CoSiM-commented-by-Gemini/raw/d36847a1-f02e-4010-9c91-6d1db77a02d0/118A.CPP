/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces (Problem 118A).
 * @details The program processes a string according to a specific set of rules:
 *          1. Delete all vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 *          2. Insert a '.' character before each consonant.
 *          3. Replace all uppercase consonants with their lowercase equivalents.
 *
 * @section algorithm
 * Algorithm: Direct String Manipulation
 * 1. An input string is read.
 * 2. The code iterates through each character of the input string.
 * 3. Each character is converted to its lowercase equivalent.
 * 4. A switch statement checks if the character is a vowel (a, o, y, e, u, i).
 * 5. If it's a vowel, the character is ignored, and the loop continues.
 * 6. If it's a consonant, a '.' character followed by the lowercase consonant
 *    is appended to a result string.
 * 7. After the loop, the final transformed string is printed.
 *
 * @section complexity
 * - Time Complexity: O(L), where L is the length of the input string. The
 *   program performs a single pass through the string.
 * - Space Complexity: O(L), to store the input and result strings.
 */
#include <iostream>
#include <string> // Required for std::string
#include <cctype> // Required for tolower

int main() {
  std::string word;
  std::string result;
  std::cin >> word;
  char c;

  /**
   * @brief Iterates through the input word to apply transformation rules.
   */
  for (int i = 0; i < word.length(); i++) {
    // Rule 3: Convert character to lowercase.
    c = tolower(word[i]);

    /**
     * @brief Rule 1: Filter out all vowels.
     * The problem defines 'y' as a vowel. If the character is a vowel,
     * the `continue` statement skips the rest of the loop body.
     */
    switch (c) {
    case 'a':
    case 'i':
    case 'u':
    case 'e':
    case 'o':
    case 'y':
      continue;
    }

    /**
     * @brief Rule 2: Prepend consonants with a '.'
     * If the character was not a vowel, a '.' and the lowercase consonant
     * are appended to the result string.
     */
    result.append(1, '.');
    result.append(1, c);
  }

  std::cout << result << std::endl;
  return 0;
}