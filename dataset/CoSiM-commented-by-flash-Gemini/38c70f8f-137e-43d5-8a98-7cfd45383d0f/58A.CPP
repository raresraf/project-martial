/**
 * @file 58A.CPP
 * @brief Solution for Codeforces Problem 58A - Chat room.
 *
 * This program determines if the word "hello" can be obtained from a given
 * input string by deleting zero or more characters. In other words, it checks
 * if "hello" is a subsequence of the input string.
 *
 * Algorithm: Two-pointer approach (subsequence matching). One pointer
 * iterates through the input string, and another pointer tracks the characters
 * of "hello". If a matching character is found, the "hello" pointer advances.
 * If all characters of "hello" are matched, the word can be formed.
 * Time Complexity: O(L_input), where L_input is the length of the input string,
 *                  as each character is visited at most once.
 * Space Complexity: O(L_input) for storing the input string (or O(1) if input is streamed).
 */

#include <iostream> // Required for standard input/output operations (cin, cout).
using namespace std; // Uses the standard namespace.

// Declare strings 'a' for input and 'b' for the target word "hello".
string a, b = "hello";
// Declare 'j' as a pointer for the target word 'b', initialized to 0.
// Declare 'pas' (short for 'passed' or 'matched') to count matched characters of 'b'.
int j = 0, pas = 0;

int main() {
  /**
   * @brief Main function to execute the Chat room problem logic.
   *
   * Reads an input string 'a'. It then attempts to find "hello" as a subsequence
   * within 'a' using a two-pointer technique. Finally, it prints "YES" if "hello"
   * is found as a subsequence, and "NO" otherwise.
   */
  // Read the input string.
  cin >> a;
  // Block Logic: Two-pointer subsequence matching.
  // Invariant: 'i' iterates through the input string 'a'. 'j' points to the
  // current character of "hello" being sought, and 'pas' counts successfully matched characters.
  for (int i = 0; i < a.size(); i++) {
    // If the current character in 'a' matches the character 'j' in "hello".
    if (a[i] == b[j]) {
      j++;   // Advance the pointer for "hello".
      pas++; // Increment the count of matched characters.

      // If all 5 characters of "hello" have been matched.
      if (pas == 5) {
        break; // Exit the loop, as "hello" has been found.
      }
    }
  }
  // Block Logic: Determine final output.
  // Pre-condition: 'pas' holds the total number of "hello" characters successfully matched.
  if (pas == 5) {
    // If all 5 characters of "hello" were matched, print "YES".
    cout << "YES";
  } else {
    // Otherwise, "hello" could not be formed as a subsequence, print "NO".
    cout << "NO";
  }
  return 0; // Indicate successful program execution.
}
