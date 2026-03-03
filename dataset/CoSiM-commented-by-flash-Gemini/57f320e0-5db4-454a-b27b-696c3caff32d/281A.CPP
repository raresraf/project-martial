/**
 * @file 281A.CPP
 * @brief Solution for Codeforces Problem 281A - "Word Capitalization".
 * This program capitalizes the first letter of a given word if it is currently lowercase.
 * If the first letter is already uppercase, it remains unchanged.
 * Algorithm: Direct character manipulation based on ASCII values.
 * Time Complexity: O(1) - as it only involves checking and potentially modifying a single character.
 * Space Complexity: O(L) - to store the input string, where L is the length of the word.
 */
#include <cstring>  // Required for string manipulation functions (though not directly used here for string objects).
#include <iostream> // Required for standard input/output operations (cin, cout).
#include <string>   // Required for std::string.

using namespace std; // Uses the standard namespace.

int main() {
  string s; // Declares a string to store the input word.
  cin >> s; // Reads the word from standard input.
  
  /**
   * Block Logic: Checks if the first character of the word is lowercase and capitalizes it if necessary.
   * Based on ASCII values: lowercase letters 'a'-'z' are 97-122. Uppercase 'A'-'Z' are 65-90.
   * The difference between a lowercase letter and its uppercase equivalent is 32.
   * Invariant: `s[0]` will be an uppercase letter after this block if it was originally lowercase.
   */
  if ((int)s[0] >= 97) // Checks if the ASCII value of the first character is in the lowercase range.
    s[0] = (int)s[0] - 32; // Converts the first character to uppercase by subtracting 32 from its ASCII value.
    
  cout << s; // Prints the modified (or original) word to standard output.
  
  return 0; // Indicates successful program execution.
}
