/**
 * @60a403d2-aca0-406b-8f97-0f9766709127/266A.CPP
 * @brief This program solves problem 266A. Given the length 'n' and a string 'str'
 * representing a sequence of elements (e.g., colored stones), it counts the number
 * of times adjacent elements in the sequence are identical.
 *
 * Algorithm: Single pass linear scan to compare adjacent characters.
 * Time Complexity: O(N) where N is the length of the input string.
 * Space Complexity: O(N) for storing the input string.
 */

#include <algorithm> // Included but not directly used in this snippet.
#include <iostream>  // For standard input/output operations (cin, cout).
#include <string>    // For using std::string.
using namespace std; // Uses the standard namespace.

int main() {
  int n, count = 0; // Declares 'n' for the length of the string and 'count' initialized to 0.
  cin >> n;         // Reads the integer 'n' (length of the string) from standard input.
  string str;       // Declares a string variable to store the input sequence.
  cin >> str;       // Reads the string 'str' from standard input.
  
  /**
   * Block Logic: Iterates through the string from the second character to compare
   * each character with its immediate predecessor.
   * Invariant: 'count' accurately reflects the number of adjacent identical characters found so far.
   */
  for (int i = 1; i < n; i++) { // Loop from i = 1 (second character) up to n-1.
    // Inline: Checks if the current character 'str[i]' is the same as the previous character 'str[i-1]'.
    if (str[i] == str[i - 1]) { 
      count++; // Increments 'count' if an adjacent identical pair is found.
    }
  }
  std::cout << count; // Prints the final count of adjacent identical characters.
  return 0;           // Indicates successful program termination.
}