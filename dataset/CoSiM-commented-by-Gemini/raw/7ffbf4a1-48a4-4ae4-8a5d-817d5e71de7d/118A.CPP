/**
 * @file 118A.CPP
 * @brief Processes a string by applying a set of transformation rules.
 *
 * The program takes a string as input and transforms it based on the following rules:
 * 1. All vowels ('A', 'O', 'Y', 'E', 'U', 'I', and their lowercase counterparts)
 *    are deleted. Note that 'Y' is considered a vowel for this problem.
 * 2. All consonants are converted to lowercase.
 * 3. A '.' character is inserted before each lowercase consonant.
 *
 * For example, "Tour" becomes ".t.r" and "aBAcAba" becomes ".b.c.b".
 *
 * @algorithm Character-by-character string traversal and transformation.
 * @complexity
 * Time: O(N), where N is the length of the input string, as we iterate through
 *       the string once.
 * Space: O(N) for the input character array. The output is streamed directly
 *        and does not require significant additional storage.
 */
#include <iostream>
#include <string>
#include <cstring> // For strlen

// Using the standard namespace for competitive programming convenience.
using namespace std;

/**
 * @brief Main function to handle string input and transformation.
 */
int main() {
  // s: a C-style character array to hold the input string.
  // The size 105 provides a buffer.
  char s[105];
  int ln, i;
  
  cin >> s;
  ln = strlen(s);
  
  // Pre-condition: 's' contains the null-terminated input string.
  // Block Logic: Iterate through each character of the string.
  for (i = 0; i < ln; i++) {
    char current_char = s[i];

    // Check if the character is an uppercase consonant.
    if (current_char >= 'A' && current_char <= 'Z' && current_char != 'A' && current_char != 'O' &&
        current_char != 'Y' && current_char != 'E' && current_char != 'U' && current_char != 'I') {
      // Convert to lowercase using ASCII offset and print in the required format.
      char c = current_char + 32;
      cout << "." << c;
    } 
    // Check if the character is a lowercase consonant.
    else if (current_char >= 'a' && current_char <= 'z' && current_char != 'a' && current_char != 'o' &&
               current_char != 'y' && current_char != 'e' && current_char != 'u' && current_char != 'i') {
      // Print in the required format.
      cout << "." << current_char;
    }
    // Implicitly, if the character is a vowel (as defined by the problem), nothing is printed.
  }
  
  cout << endl;
  return 0;
}
