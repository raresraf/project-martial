/**
 * @68f92dd6-d799-4f29-b755-f161b0e57103/58A.CPP
 * @brief This program solves Codeforces problem 58A, "Chat room".
 * It determines if the word "hello" can be formed as a subsequence from a
 * given input string. A subsequence does not require consecutive characters,
 * but they must appear in the correct order.
 *
 * Algorithm: Two-pointer approach. One pointer iterates through the input string,
 * and another pointer tracks the characters of "hello". If a character from
 * "hello" is found in the input string, the "hello" pointer advances.
 * Time Complexity: O(L) where L is the length of the input string, as it involves a single pass.
 * Space Complexity: O(1) beyond input string storage (the target word "hello" is constant size).
 */
#include <iostream> // For standard input/output operations (cin, cout).
#include <string>   // For using std::string.
using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to check if "hello" is a subsequence of an input string.
 * Reads an input string, uses a two-pointer approach to find "hello" as a subsequence,
 * and prints "YES" or "NO".
 * @return 0 on successful execution.
 */
int main() {
  string s; // Declares a string variable to store the input string.
  cin >> s; // Reads the input string from standard input.

  // Declares a character array representing the target word "hello".
  char a[] = {'h', 'e', 'l', 'l', 'o'}; 
  int j = 0; // Pointer/index for the characters of the target word `a`.

  /**
   * Block Logic: Iterates through the input string `s` to find the characters of "hello" as a subsequence.
   * If a character in `s` matches the current target character in `a` (pointed to by `j`),
   * then `j` is incremented, moving to the next character of "hello".
   * Invariant: `j` counts how many characters of "hello" have been found in order as a subsequence.
   */
  for (int i = 0; i < s.length(); i++) { // Loops through each character of the input string.
    // Pre-condition: Checks if the current character in `s` matches the target character in `a`.
    // Also ensures `j` is within bounds of 'a' array (implied by `j++` checking).
    if (s[i] == a[j]) { 
      j++; // Advances the pointer for "hello" to find the next character.
    }
  }

  /**
   * Block Logic: Determines if all characters of "hello" were found as a subsequence.
   * If `j` reached 5 (the length of "hello"), then all characters were found.
   * Pre-condition: `j` holds the count of found "hello" characters.
   * Invariant: Prints "YES" if "hello" is a subsequence, "NO" otherwise.
   */
  if (j < 5) // If `j` is less than 5, not all characters of "hello" were found.
    cout << "NO"; // Prints "NO".
  else { // If `j` is 5, all characters of "hello" were found.
    cout << "YES"; // Prints "YES".
  }
  return 0; // Indicates successful program termination.
}