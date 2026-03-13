/**
 * @file 58A.CPP
 * @brief Checks if the string "hello" is a subsequence of an input string.
 *
 * This program reads a string and determines if the word "hello" can be obtained
 * by deleting zero or more characters from the input.
 *
 * Algorithm:
 * 1. Define the target subsequence, "hello".
 * 2. Initialize a pointer 'a' to the start of the target subsequence (index 0).
 * 3. Iterate through the input string character by character.
 * 4. If the current character of the input string matches the character at
 *    the current subsequence pointer 'a', advance the pointer 'a'.
 * 5. After iterating through the entire input string, if the pointer 'a' has
 *    advanced past all 5 characters of "hello", it means the subsequence was found.
 * 6. Print "YES" if the subsequence is found, otherwise print "NO".
 *
 * Time Complexity: O(L), where L is the length of the input string, because it
 * requires a single pass.
 * Space Complexity: O(L) to store the input string.
 */
#include <iostream>

using namespace std;

int main() {
  string hello = "hello"; // The target subsequence.
  int a = 0;              // Pointer for the current character in the 'hello' string.
  int c = 0;              // Counter for matched characters (redundant, 'a' is sufficient).
  string palabra;         // The input string to search within.
  
  cin >> palabra;

  /**
   * Block Logic: Iterate through the input string to find the 'hello' subsequence.
   * Invariant: 'a' represents the length of the prefix of "hello" that has been
   * matched so far.
   */
  for (int i = 0; i < palabra.size(); i++) {
    // If the current character in the input matches the character we're looking for...
    if (palabra[i] == hello[a]) {
      a++; // ...move to the next character in the 'hello' target.
      c++; // Increment the match counter.
    }
  }
  
  // Check if all 5 characters of "hello" were found in order.
  (c == 5) ? cout << "YES" : cout << "NO";
  
  return 0;
}
