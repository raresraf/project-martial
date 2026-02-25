/**
 * @file 281A.CPP
 * @brief Solution to Codeforces problem 281A (Word Capitalization).
 * @author Solved by: [User who submitted the code]
 *
 * @details
 * This program takes a single word as input and capitalizes its first letter.
 * All other letters in the word remain unchanged.
 *
 * @section problem_description
 * Capitalization is writing a word with its first letter as a capital letter.
 * Your task is to capitalize the given word. Note, that during capitalization
 * all the letters except the first one remains unchanged.
 * Input: A single line contains a non-empty word. This word consists of
 * lowercase and uppercase English letters. The length of the word will not
 * exceed 10^3.
 * Output: Output the given word after capitalization.
 *
 * @section algorithm
 * The algorithm is straightforward:
 * 1. Read the input word into a string.
 * 2. Access the first character of the string (at index 0).
 * 3. Convert this character to its uppercase equivalent using the `toupper()`
 *    function.
 * 4. The original string is modified in-place.
 * 5. Print the modified string.
 *
 * @section complexity
 * - Time Complexity: O(L), where L is the length of the string, due to input/output
 *   operations. The capitalization itself is an O(1) operation.
 * - Space Complexity: O(L) to store the word in a string.
 */

#include <cctype>   // For toupper()
#include <iostream> // For cin, cout
#include <string>   // For std::string

using namespace std;

int main() {
  string s;
  // Read the word from standard input.
  cin >> s;

  // Capitalize the first letter of the string.
  // The toupper() function converts a lowercase letter to uppercase.
  // If the character is already uppercase or not a letter, it remains unchanged.
  s[0] = toupper(s[0]);

  // Print the modified string to standard output, followed by a newline.
  cout << s << endl;

  return 0;
}
