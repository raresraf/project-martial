/**
 * @file 58A.CPP
 * @brief Checks if the word "hello" can be formed as a subsequence from a given input string.
 *        A subsequence does not require consecutive characters, but they must maintain
 *        their relative order.
 * Algorithm: Sequential subsequence search. Iterates through the input string,
 *            searching for the characters 'h', 'e', 'l', 'l', 'o' in order.
 *            The search for each subsequent character starts from the position
 *            immediately after the previous character was found.
 * Time Complexity: O(L) where L is the length of the input string, as each character
 *                  is visited at most a constant number of times (five searches).
 * Space Complexity: O(L) for storing the input string.
 */
#include <iostream> // For standard input/output operations.
#include <string>   // For string manipulation.

// Using declarations to simplify code by avoiding std:: prefix.
using std::cin;
using std::cout;
using std::endl;
using std::string;

/**
 * @brief Main function to read an input string and determine if "hello" is its subsequence.
 * Functional Utility: Provides a solution to identify if a specific pattern (subsequence)
 *                     exists within a longer string, preserving relative order.
 */
int main() {
  string input; // Declares a string variable to hold the input string.
  cin >> input; // Reads the input string from standard input.

  size_t pos = 0; // Current position in the input string to start searching from.
                   // `size_t` is an unsigned type appropriate for sizes and indices.
  
  // Block Logic: Searches for the character 'h' in the input string.
  // Invariant: `pos` advances until 'h' is found or the end of the string is reached.
  while (pos < input.length() && input[pos] != 'h') {
    ++pos; // Move to the next character.
  }
  ++pos; // Move past 'h' to search for the next character.
  
  // Block Logic: Searches for the character 'e' from the current `pos`.
  // Invariant: `pos` advances until 'e' is found or the end of the string is reached.
  while (pos < input.length() && input[pos] != 'e') {
    ++pos;
  }
  ++pos; // Move past 'e'.
  
  // Block Logic: Searches for the first 'l' from the current `pos`.
  // Invariant: `pos` advances until 'l' is found or the end of the string is reached.
  while (pos < input.length() && input[pos] != 'l') {
    ++pos;
  }
  ++pos; // Move past the first 'l'.
  
  // Block Logic: Searches for the second 'l' from the current `pos`.
  // Invariant: `pos` advances until the second 'l' is found or the end of the string is reached.
  while (pos < input.length() && input[pos] != 'l') {
    ++pos;
  }
  ++pos; // Move past the second 'l'.
  
  // Block Logic: Searches for the character 'o' from the current `pos`.
  // Invariant: `pos` advances until 'o' is found or the end of the string is reached.
  while (pos < input.length() && input[pos] != 'o') {
    ++pos;
  }

  // Functional Utility: Checks if 'o' was successfully found within the string.
  //                     If `pos` is still within the bounds of the string, it means 'o' was found.
  if (pos < input.length()) {
    cout << "YES"; // Prints "YES" if "hello" is a subsequence.
  } else {
    cout << "NO";  // Prints "NO" otherwise.
  }
  
  return 0; // Indicates successful program termination.
}