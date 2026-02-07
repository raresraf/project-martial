/**
 * @file 281A.CPP
 * @brief Solution to a competitive programming problem involving string manipulation.
 * This program reads a single word (string) and capitalizes its first letter
 * if it is initially a lowercase English alphabet character. The modified string
 * is then printed to standard output.
 *
 * Algorithm: Character case conversion.
 * Time Complexity: O(1) as it only checks and potentially modifies the first character.
 * Space Complexity: O(L) for storing the input string, where L is its length.
 */
#include <iostream>
using namespace std;
/**
 * @brief Main function of the program.
 * Reads a single word from standard input, capitalizes its first letter if it's
 * lowercase, and then prints the modified word to standard output.
 * @return `0` upon successful execution.
 */
int main() {
  char s[1000]; // Declares a character array of size 1000 to store the input word.
  cin >> s; // Reads a word from standard input into the character array `s`.
  /**
   * @brief Conditional logic for capitalizing the first letter.
   * If the first character of the string `s` is a lowercase English alphabet letter,
   * its ASCII value is reduced by 32 to convert it to its uppercase equivalent.
   */
  if (s[0] >= 'a' && s[0] <= 'z') {
    s[0] = s[0] - 32;
  }
  cout << s; // Prints the modified string to standard output.
  return 0;
}
