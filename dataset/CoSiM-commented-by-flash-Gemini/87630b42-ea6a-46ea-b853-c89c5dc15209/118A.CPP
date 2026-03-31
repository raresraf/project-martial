/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces (Problem 118A).
 * This program transforms an input string by converting uppercase letters to lowercase,
 * deleting all vowels ('a', 'o', 'y', 'e', 'u', 'i'), and inserting a "." before each consonant.
 *
 * Algorithm: Character-by-character processing. Each character is checked explicitly
 * for being a lowercase or uppercase consonant. Vowels are skipped. Consonants are
 * printed with a preceding dot, converted to lowercase if originally uppercase.
 * Time Complexity: O(L), where L is the length of the input string, as each character is processed once.
 * Space Complexity: O(L) for storing the input string.
 */

#include <cstring>  // @brief Required for string manipulation functions (not directly used but common).
#include <iostream> // @brief Required for standard input/output streams (cin, cout).
#include <cctype>   // @brief Required for std::tolower.
using namespace std;
int main() {
  /**
   * @brief Main function to perform the string transformation task.
   * Reads strings from standard input until EOF, applies transformation rules
   * (lowercase, remove vowels, add '.' before consonants), and prints the result.
   */
  int i, j, k, l; // @brief Loop counters and temporary variables (j, k, l are unused in this snippet).
  string s; // @brief The input string.
  
  // Block Logic: Read strings from standard input until EOF.
  // Invariant: Each string is processed according to the rules before the next is read.
  while (cin >> s) {
    l = s.length(); // @brief Length of the current input string.
    
    // Block Logic: Iterate through each character of the input string to apply transformations.
    // Pre-condition: 's' contains the string to be transformed.
    // Invariant: Each character from 's' is processed, and the transformed output is printed.
    for (i = 0; i < l; i++) {
      // Block Logic: Check if the character is a lowercase consonant.
      // If it's a lowercase letter and not one of the vowels ('a', 'e', 'i', 'o', 'u', 'y').
      if (s[i] >= 'a' && s[i] <= 'z' && s[i] != 'a' && s[i] != 'e' &&
          s[i] != 'i' && s[i] != 'o' && s[i] != 'y' && s[i] != 'u') {
        cout << "."; // Inline: Print a '.'
        cout << s[i]; // Inline: Print the lowercase consonant.
      } 
      // Block Logic: Check if the character is an uppercase consonant.
      // If it's an uppercase letter and not one of the vowels ('A', 'E', 'I', 'O', 'U', 'Y').
      else if (s[i] >= 'A' && s[i] <= 'Z' && s[i] != 'A' && s[i] != 'E' &&
                 s[i] != 'I' && s[i] != 'O' && s[i] != 'Y' && s[i] != 'U') {
        cout << "."; // Inline: Print a '.'
        cout << (char)tolower(s[i]); // Inline: Print the lowercase version of the uppercase consonant.
      }
    }
    cout << endl; // @brief Print a newline character after processing each string.
  }
  return 0; // @brief Indicates successful program termination.
}
