/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces (Problem 118A).
 * This program transforms an input string by converting uppercase letters to lowercase,
 * deleting all vowels ('a', 'o', 'y', 'e', 'u', 'i'), and inserting a "." before each consonant.
 *
 * Algorithm: Character-by-character processing. Each character is checked for being a vowel
 * or a consonant, converted to lowercase, and then printed with a preceding dot if it's a consonant.
 * Time Complexity: O(L), where L is the length of the input string, as each character is processed once.
 * Space Complexity: O(1) beyond input/output storage, as processing is done in-place per character.
 */

#include <iostream> // @brief Required for standard input/output streams (cin, cout).
#include <string>   // @brief Required for std::string.
#include <cctype>   // @brief Required for std::tolower.
using namespace std;

int main() {
  /**
   * @brief Main function to perform the string transformation task.
   * Reads an input string, iterates through its characters, and prints a new string
   * according to the transformation rules (lowercase consonants preceded by '.').
   */
  int i; // @brief Loop counter for iterating through the string.
  string s; // @brief The input string.
  cin >> s;
  
  // Block Logic: Iterate through each character of the input string to apply transformations.
  // Pre-condition: 's' contains the input string.
  // Invariant: Each character from 's' is processed, and the transformed output is printed.
  for (i = 0; i < s.length(); i++) {
    char x = tolower(s[i]); // Inline: Convert the current character to lowercase.
    
    // Block Logic: Check if the character is a vowel (including 'y').
    // If it's a vowel, skip it (continue to the next character).
    if (x == 'a' || x == 'e' || x == 'i' || x == 'o' || x == 'u' || x == 'y') {
      continue;
    } 
    // Otherwise, it's a consonant, so print a "." followed by the lowercase character.
    else
      cout << "." << x;
  }
  return 0; // @brief Indicates successful program termination.
}