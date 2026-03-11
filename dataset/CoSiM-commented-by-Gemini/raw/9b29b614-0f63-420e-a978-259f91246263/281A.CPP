/**
 * @file 281A.CPP
 * @brief Capitalizes the first letter of a given input string.
 *
 * This program reads a single line of text from standard input. It then
 * checks if the first character of the string is a lowercase English
 * alphabet letter. If it is, the character is converted to its uppercase
 * equivalent. Finally, the modified string is printed to standard output.
 * This is a common requirement in string manipulation tasks where sentences
 * or names must start with a capital letter.
 *
 * @algorithm The core logic involves a direct ASCII value manipulation.
 * By subtracting 32 from the ASCII value of a lowercase letter, we get the
 * ASCII value of its corresponding uppercase letter.
 *
 * @time_complexity O(1) because it only inspects and modifies the first character.
 * The input reading is O(N) where N is the length of the string, but the main logic is constant time.
 * For the purpose of the problem, we can consider it O(1) as the operation is on a fixed position.
 * @space_complexity O(N) for storing the input string, where N is the length of the string.
 */
#include <iostream>

using namespace std;

int main() {

  string s = "";

  // Read a full line of text from standard input into the string 's'.
  getline(cin, s);

  /**
   * @brief Block to capitalize the first character if it's a lowercase letter.
   * @invariant The string 's' contains the user's input.
   * @pre 's' is not empty. The behavior is undefined for an empty string, though in C++ `s[0]` on an empty string is undefined behavior.
   * However, given the competitive programming context, we can assume valid inputs.
   */
  if (s[0] >= 97 && s[0] <= 122) { // Checks if the first character is a lowercase letter (ASCII 'a' to 'z').

    // Convert the lowercase first character to uppercase by adjusting its ASCII value.
    s[0] = s[0] - 32;
  }

  // Print the modified (or original) string to standard output.
  cout << s;

  return 0;
}
