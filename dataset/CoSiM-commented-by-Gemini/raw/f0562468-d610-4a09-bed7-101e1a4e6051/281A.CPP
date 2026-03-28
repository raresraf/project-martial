/**
 * @file 281A.CPP
 * @brief Capitalizes the first letter of a word.
 *
 * This program reads a single word and capitalizes its first character if it's
 * a lowercase letter.
 *
 * @b Algorithm: The program reads a string and checks the ASCII value of its
 *    first character. If the value is 97 ('a') or greater, it's considered a
 *    lowercase letter and is converted to uppercase by subtracting 32 from its
 *    ASCII value.
 *
 * @b Time_Complexity: O(L), where L is the length of the input string, due to
 *    the I/O operations. The capitalization itself is an O(1) operation.
 * @b Space_Complexity: O(L) to store the input string.
 */
#include <iostream>
#include <string.h>
using namespace std;
int main() {
  string word;
  cin >> word;

  /**
   * @brief This block checks if the first character is lowercase and converts it.
   * In ASCII, lowercase letters start at 97 ('a'). If the character's value
   * is >= 97, it's a lowercase letter. Subtracting 32 converts it to the
   * corresponding uppercase letter (e.g., 'a' (97) - 32 = 'A' (65)).
   */
  if (int(word[0]) >= 97) {
    word[0] = char(int(word[0]) - 32);
  }
  cout << word << endl;
  return 0;
}