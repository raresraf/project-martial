/**
 * @file 281A.CPP
 * @brief Solution to a programming problem that capitalizes the first letter of a word.
 * @details This program reads a single word and ensures that its first character is uppercase.
 * If the first character is already uppercase or not a letter, it remains unchanged.
 *
 * Algorithm:
 * 1. Read the input string `kata`.
 * 2. Check if the first character `kata[0]` is a lowercase English letter (i.e., between 'a' and 'z').
 * 3. If it is, convert it to its uppercase equivalent. This is achieved using ASCII arithmetic:
 *    subtracting the ASCII value of 'a' and adding the ASCII value of 'A' effectively shifts the character
 *    from the lowercase to the uppercase range.
 * 4. Print the resulting string, which will have its first letter capitalized.
 *
 * Time Complexity: O(N), where N is the length of the string, because of the I/O operations. The capitalization itself is O(1).
 * Space Complexity: O(N), to store the string `kata`.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  string kata;
  cin >> kata;

  /**
   * Block Logic: Check if the first character is lowercase and capitalize it if so.
   * Pre-condition: 'kata' holds the input word.
   * Invariant: The operation only modifies the first character if it's in the range 'a' through 'z'.
   */
  if ((kata[0] >= 'a') && (kata[0] <= 'z')) {
    // Convert the first character to uppercase using ASCII value manipulation.
    kata[0] = kata[0] - 'a' + 'A';
  }
  
  // Print the modified or original string.
  cout << kata;

  return 0;
}
