/**
 * @file 281A.CPP
 * @brief Solution for a string capitalization problem (Word Capitalization).
 *
 * @details This program reads a single word from standard input and ensures its
 * first letter is capitalized. If the first letter is already uppercase or not a
 * letter, the word remains unchanged.
 *
 * @section algorithm
 * Algorithm:
 * 1. Read the input word into a string variable.
 * 2. Check if the first character is a lowercase English letter by comparing its
 *    ASCII value.
 * 3. If it is lowercase, convert it to uppercase using ASCII arithmetic
 *    (subtracting 32).
 * 4. Update the first character of the string with the new uppercase character.
 * 5. Print the resulting string.
 *
 * @section complexity
 * Time Complexity: O(L), where L is the length of the input word. This is
 * dominated by the I/O operations of reading and printing the string. The
 * capitalization logic itself is O(1).
 * Space Complexity: O(L), to store the input word.
 */
#include <iostream>

using namespace std;

int main() {
  string str;
  cin >> str;

  // Block Logic: Capitalize the first letter if it is lowercase.
  // Pre-condition: The input string 'str' is not empty.
  // The ASCII value of 'a' is 97. This checks if the first character is in the
  // range 'a' through 'z'.
  if ((int)str.at(0) >= 97) {
    // Inline: Convert from lowercase to uppercase by subtracting 32 from the
    // ASCII value. This is the difference between a lowercase and uppercase
    // letter in the ASCII table (e.g., 'a' (97) - 'A' (65) = 32).
    int ascii = str.at(0) - 32;
    char asciiToChar = static_cast<char>(ascii);
    str.at(0) = asciiToChar;
    cout << str << endl;
  } else {
    // If the first character is already uppercase or not a letter, print as is.
    cout << str << endl;
  }
  return 0;
}
