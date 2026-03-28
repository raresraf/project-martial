/**
 * @file 281A.CPP
 * @brief Capitalizes the first letter of a word.
 *
 * This program reads a single word and capitalizes its first character.
 *
 * @b Algorithm: The program reads a string, accesses its first character at
 *    index 0, and converts it to uppercase using `std::toupper`.
 *
 * @b Time_Complexity: O(L), where L is the length of the input string, due to
 *    the I/O operations. The capitalization itself is an O(1) operation.
 * @b Space_Complexity: O(L) to store the input string.
 */
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  // Disables synchronization with C-style I/O for faster input/output.
  std::ios_base::sync_with_stdio(0);

  std::string s;
  std::cin >> s;

  // Capitalize the first character of the string.
  s[0] = std::toupper(s[0]);

  std::cout << s << std::endl;
  return 0;
}
