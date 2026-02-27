/**
 * @file 281A.CPP
 * @brief Solution for competitive programming problem: "Word Capitalization".
 *
 * @details This program takes a single word as input and capitalizes its first
 *          letter, leaving the rest of the word unchanged. The transformed word
 *          is then printed to standard output.
 *          Algorithm: Direct character manipulation for the first letter, followed
 *                     by printing the rest of the string as is.
 *          Time Complexity: O(L), where L is the length of the input word, as
 *                           each character is processed or printed once.
 *          Domain: Competitive Programming.
 */
#include <ctype.h>
#include <iostream>
#include <string>
using namespace std;

int main() {
  string str; // Declare string 'str' to store the input word.
  cin >> str;  // Read the input word.
  
  // Block Logic: Capitalize and print the first character of the string.
  // `toupper` converts a character to its uppercase equivalent.
  // `putchar` prints a single character to standard output.
  putchar(toupper(str[0]));
  
  // Block Logic: Iterate through the rest of the string (from the second character) and print each character as is.
  for (int i = 1; i < str.length(); i++) {
    std::cout << str[i]; // Print the character.
  }
  // The original code did not have an endl here.
  return 0; // Indicate successful program termination.
}