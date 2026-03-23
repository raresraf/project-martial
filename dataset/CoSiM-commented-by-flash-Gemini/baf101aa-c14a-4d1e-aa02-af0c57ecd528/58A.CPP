/**
 * @file 58A.CPP
 * @brief This competitive programming solution determines if the word "hello"
 *        can be formed as a subsequence from a given input string.
 *        A subsequence implies that the characters must appear in the correct order,
 *        but not necessarily consecutively.
 * Algorithm: Sequential Subsequence Search.
 *            The algorithm iterates through the input string, searching for each
 *            character of the target word "hello" in its correct sequence.
 *            For 'h', 'e', 'l', 'l', 'o', it finds the first occurrence of the
 *            current character starting from the position immediately after
 *            the previous character was found.
 * Time Complexity: O(L), where L is the length of the input string. This is because
 *                  each character of the input string is visited at most a constant
 *                  number of times (specifically, it makes at most five passes, one for each
 *                  letter in "hello", through the remaining part of the string).
 * Space Complexity: O(L) for storing the input string.
 */
#include <iostream> // Required for standard input/output operations (cin, cout, endl).
#include <string>   // Required for string manipulation (std::string).
using namespace std;

/**
 * @brief Main function to read an input string and determine if "hello" is its subsequence.
 * Functional Utility: Serves as the entry point for the "Chat room" problem,
 *                     implementing the core logic for subsequence detection.
 */
int main() {
  string s;   // Declares a string variable to hold the input string.
  cin >> s;   // Reads the input string from standard input.
  size_t pos = 0; // Current position in the input string to start searching from.

  // Block Logic: Searches for the character 'h' in the input string 's'.
  // Invariant: `pos` is advanced until 'h' is found or the end of the string is reached.
  while (pos < s.length() && s[pos] != 'h')
    ++pos; // Move to the next character in 's'.
  ++pos;   // Move past 'h' to search for the next character ('e') after this 'h'.

  // Block Logic: Searches for the character 'e' in the input string 's', starting from `pos`.
  // Invariant: `pos` is advanced until 'e' is found or the end of the string is reached.
  while (pos < s.length() && s[pos] != 'e')
    ++pos; // Move to the next character.
  ++pos;   // Move past 'e'.

  // Block Logic: Searches for the first 'l' in the input string 's', starting from `pos`.
  // Invariant: `pos` is advanced until 'l' is found or the end of the string is reached.
  while (pos < s.length() && s[pos] != 'l')
    ++pos; // Move to the next character.
  ++pos;   // Move past the first 'l'.

  // Block Logic: Searches for the second 'l' in the input string 's', starting from `pos`.
  // Invariant: `pos` is advanced until the second 'l' is found or the end of the string is reached.
  while (pos < s.length() && s[pos] != 'l')
    ++pos; // Move to the next character.
  ++pos;   // Move past the second 'l'.

  // Block Logic: Searches for the character 'o' in the input string 's', starting from `pos`.
  // Invariant: `pos` is advanced until 'o' is found or the end of the string is reached.
  while (pos < s.length() && s[pos] != 'o')
    ++pos; // Move to the next character.

  // Functional Utility: Checks if 'o' was successfully found within the string.
  //                     If `pos` is still within the bounds of the string, it means 'o' was found,
  //                     and therefore "hello" is a subsequence.
  if (pos < s.length()) {
    cout << "YES" << endl; // Prints "YES" if "hello" is a subsequence.
  } else {
    cout << "NO" << endl;  // Prints "NO" otherwise.
  }
  return 0; // Indicates successful program termination.
}
