/**
 * @6694664f-b659-4529-9fed-08757baf9d80/281A.CPP
 * @brief This program solves the "Word Capitalization" problem (Codeforces 281A).
 * It reads a single word, capitalizes its first letter, and then prints the
 * modified word to standard output.
 *
 * Algorithm: String manipulation. Reads a string, converts its first character
 * to uppercase, and then prints the modified string.
 * Time Complexity: O(L) where L is the length of the input string.
 * Space Complexity: O(L) for storing the input string.
 */
#include <bits/stdc++.h> // Includes many standard C++ library headers, including <iostream> and <string>.
using namespace std;     // Uses the standard namespace.

/**
 * @brief Main function to capitalize the first letter of an input word.
 * Reads a word, modifies its first character to uppercase, and prints the result.
 * @return 0 on successful execution.
 */
int main() {
  string str; // Declares a string variable to store the input word.
  char ch;    // Declares a char variable to store the capitalized first character.
  int n;      // Stores the length of the input string.
  int i;      // Loop counter.
  
  cin >> str; // Reads the word from standard input into `str`.
  n = str.size(); // Gets the length of the string.
  ch = toupper(str[0]); // Inline: Converts the first character of the string to uppercase.
  cout << ch; // Prints the capitalized first character.
  
  /**
   * Block Logic: Iterates through the rest of the string (from the second character)
   * and prints each character as is.
   * Invariant: All characters from the second one onwards are printed to output.
   */
  for (i = 1; i < n; i++) { // Loop starts from the second character (index 1).
    cout << str[i]; // Prints the current character.
  }
  return 0; // Indicates successful program termination.
}
