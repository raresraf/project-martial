/**
 * @file 118A.CPP
 * @brief Solution to Codeforces problem 118A (String Task).
 * @author Solved by: [User who submitted the code]
 *
 * @details
 * This program processes an input string based on a specific set of rules:
 * 1. All uppercase letters are converted to lowercase.
 * 2. All vowels ('a', 'o', 'y', 'e', 'u', 'i') are deleted.
 * 3. A '.' character is inserted before every consonant.
 *
 * The final modified string is printed to the console.
 *
 * @section algorithm
 * The algorithm iterates through the input string character by character in a
 * single pass:
 * 1. Read the entire string from standard input.
 * 2. For each character in the string:
 *    a. Convert the character to its lowercase equivalent.
 *    b. Check if the lowercase character is a vowel (a, o, y, e, u, i).
 *    c. If it is a vowel, do nothing and proceed to the next character (effectively
 *       deleting it).
 *    d. If it is a consonant, print a '.' character followed by the lowercase
 *       consonant.
 *
 * This approach builds the output string on the fly without creating an
 * intermediate string, making it memory-efficient.
 *
 * @section complexity
 * - Time Complexity: O(L), where L is the length of the input string, as it
 *   involves a single pass through the string.
 * - Space Complexity: O(L) to store the input string. The output is streamed
 *   directly, so additional space is minimal.
 */

#include <iostream>
#include <string>
#include <cctype> // For tolower()
using namespace std;

int main() {
  string s;
  cin >> s;

  /**
   * @brief Loop through each character of the input string.
   */
  for (int i = 0; i < s.length(); i++) {
    // Convert character to lowercase for case-insensitive vowel checking.
    char c = tolower(s[i]);

    // Check if the character is a vowel.
    if (c == 'a' || c == 'o' || c == 'y' || c == 'e' || c == 'u' || c == 'i') {
      // If it's a vowel, skip it (effectively deleting it).
      continue;
    } else {
      // If it's a consonant, print a '.' followed by the lowercase consonant.
      cout << "." << c;
    }
  }
  cout << endl; // Add a final newline for good practice.
  return 0;
}
