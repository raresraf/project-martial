/**
 * @file 281A.CPP
 * @brief Solution for Codeforces problem 281A, "Word Capitalization".
 *
 * @description
 * This program takes a single word as input and capitalizes its first letter
 * if it is lowercase. The rest of the word remains unchanged.
 *
 * @algorithm
 * 1. Read the input word into a character array.
 * 2. Check if the first character (at index 0) is a lowercase letter
 *    (between 'a' and 'z').
 * 3. If it is, convert it to its uppercase equivalent by manipulating its
 *    ASCII value.
 * 4. Print the modified word.
 *
 * @time_complexity O(L) to read and print the string, where L is the length
 *   of the word. The capitalization itself is an O(1) operation.
 * @space_complexity O(L) to store the word in a character array.
 */
#include <cstring>
#include <iostream>

int main() {
  // Using a C-style character array to store the string.
  char str[1005];
  std::cin >> str;

  // Block Logic: Check if the first character is a lowercase letter.
  if ((str[0] <= 'z') && (str[0] >= 'a'))
    // Convert to uppercase by subtracting 32 from the ASCII value.
    // e.g., ASCII('a') = 97, ASCII('A') = 65. 97 - 32 = 65.
    str[0] = (char)((int)str[0] - 32);
  
  std::cout << str << '\n';
  return 0;
}
