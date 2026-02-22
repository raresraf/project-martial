/**
 * @file 281A.CPP
 * @brief An efficient solution to the "Word Capitalization" problem (Codeforces 281A).
 *
 * @algorithm
 * The problem requires capitalizing the first letter of a given word.
 *
 * This solution uses an efficient "print-as-you-go" approach that avoids
 * modifying the original string or creating a new one.
 * 1. Read the input word into a string.
 * 2. Take the first character of the string, convert it to uppercase, and
 *    print it immediately.
 * 3. Loop from the second character to the end of the string.
 * 4. Print each of these characters unchanged.
 *
 * This method is efficient as it minimizes string manipulation and memory operations.
 *
 * @complexity
 * - Time Complexity: O(L), where L is the length of the string, for reading and
 *   then printing the characters once.
 * - Space Complexity: O(L) to store the input string.
 */
#include <ctype.h>
#include <iostream>
#include <string>
using namespace std;

int main() {
  string str;
  cin >> str;
  /**
   * @brief Capitalize and print the first character.
   * This uses `toupper` to convert the character and `putchar` (a C-style
   * function) to print it to standard output.
   */
  putchar(toupper(str[0]));
  /**
   * @brief Loop from the second character to the end of the string and print
   * the remainder of the word unchanged.
   */
  for (int i = 1; i < str.length(); i++) {
    std::cout << str[i];
  }
  return 0;
}