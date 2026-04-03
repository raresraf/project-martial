/**
 * @file 281A.cpp
 * @brief Solution to a competitive programming problem: Word Capitalization.
 * This program reads a word from the standard input, capitalizes its first letter
 * if it is a lowercase letter, and then prints the modified word to the standard output.
 *
 * Algorithm:
 * 1. Read a line of text (a single word in this case) from the input.
 * 2. Check if the first character of the word is a lowercase English letter.
 * 3. If it is, convert it to its corresponding uppercase letter by subtracting 32 from its ASCII value.
 * 4. Print the resulting string.
 *
 * Time Complexity: O(1) as it only operates on the first character of the string.
 * Space Complexity: O(L) where L is the length of the input word.
 */

#include <bits/stdc++.h>

int main() {
  // A string variable to store the input word.
  std::string str;
  
  // Read a line from standard input into the string.
  std::getline(std::cin, str);
  
  // Check if the first character is a lowercase letter.
  if (std::islower(str[0]))
    // Convert the first character to uppercase by subtracting the ASCII difference.
    str[0] -= 32;
  
  // Print the modified string to the console, followed by a newline.
  printf("%s\n", str.c_str());

  return 0;
}
