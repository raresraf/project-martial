/**
 * @file 281A.CPP
 * @brief Solution to a string manipulation problem. The program reads a single
 *        word and capitalizes its first letter if it is currently lowercase.
 *        The rest of the word remains unchanged.
 *
 * Algorithm: Direct character modification based on ASCII values.
 *            The algorithm directly accesses the first character of the input
 *            string. It checks if its ASCII value falls within the range of
 *            lowercase letters. If so, it subtracts 32 (the difference between
 *            ASCII values of 'a' and 'A') to convert it to its uppercase equivalent.
 * Time Complexity: O(1), as only the first character of the string is accessed
 *                  and modified.
 * Space Complexity: O(L) for storing the input string, where L is the length of the string.
 */

#include <cctype>    // For character manipulation functions (though not explicitly used in the final version, good practice)
#include <iostream>  // For input/output operations (std::cin, std::cout)
#include <string>    // For std::string class

int main() {
  std::string slowo; // Declares a string variable to store the input word.
  // Block Logic: Reads a single word from standard input into the `slowo` variable.
  std::cin >> slowo;
  // Pre-condition: Checks if the ASCII value of the first character is greater than 90.
  // This condition efficiently determines if the character is a lowercase letter,
  // as uppercase letters have ASCII values <= 90 and lowercase letters start from 97.
  if (slowo[0] > 90) { // ASCII value for 'Z' is 90. 'a' is 97.
    // Functional Utility: Converts the first character to uppercase by subtracting 32 from its ASCII value.
    // This is equivalent to `slowo[0] = toupper(slowo[0]);` but uses direct ASCII manipulation.
    slowo[0] -= 32;
  }
  // Post-condition: Prints the modified word (with the first letter potentially capitalized)
  // to standard output.
  std::cout << slowo;

  return 0; // Indicates successful program execution.
}