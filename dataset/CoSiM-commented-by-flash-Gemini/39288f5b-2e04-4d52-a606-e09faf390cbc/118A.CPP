/**
 * @file 118A.CPP
 * @brief Solution for Codeforces Problem 118A - String Task.
 *
 * This program transforms an input string according to specific rules:
 * 1. All uppercase letters are converted to their lowercase equivalents.
 * 2. All vowels ('a', 'o', 'y', 'e', 'u', 'i') are deleted.
 * 3. A dot ('.') is inserted before each consonant.
 *
 * Algorithm: Character iteration and conditional formatting. The program iterates
 * through each character of the input string, converts it to lowercase, and then
 * applies conditional logic to determine whether to print a dot followed by the
 * character (if it's a consonant) or skip it (if it's a vowel).
 * Time Complexity: O(L), where L is the length of the input string, as each character
 *                  is processed once.
 * Space Complexity: O(L) for storing the input string (or O(1) if output is streamed).
 */

#include <ctype.h>  // Required for the tolower() function.
#include <iostream> // Required for standard input/output operations (cin, cout).

using namespace std; // Uses the standard namespace.

int main() {
  /**
   * @brief Main function to execute the String Task problem logic.
   *
   * Reads an input string. It then iterates through each character of the string,
   * converts it to lowercase, and conditionally prints it with a preceding dot
   * if it is a consonant, effectively deleting vowels.
   */
  // Declare a character array 's' to store the input string (max 100 characters + null terminator).
  char s[101];
  // Declare 'aux' (unused in this solution, possibly leftover from template/draft).
  char aux;
  // Declare 'i' as a loop counter/index, initialized to 0.
  int i = 0;

  // Read the input string into the character array 's'.
  cin >> s;

  // Block Logic: Iterate through the string, applying transformation rules.
  // Invariant: 'i' advances through the characters of the string until the null terminator is reached.
  while (s[i]) { // Loop continues as long as s[i] is not the null terminator.
    // Convert the current character to lowercase.
    s[i] = tolower(s[i]);

    // Conditional Logic: Check if the current character is a vowel.
    // The vowels are defined as 'a', 'e', 'i', 'o', 'u', and 'y'.
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y') {
      i++;     // If it's a vowel, simply advance to the next character (effectively deleting it).
      continue; // Skip the rest of the loop body for this iteration.
    } else {
      // If it's a consonant, print a dot followed by the consonant.
      cout << "." << s[i];
      i++; // Advance to the next character.
    }
  }

  // Print a newline character at the end of the output.
  cout << endl;

  return 0; // Indicate successful program execution.
}