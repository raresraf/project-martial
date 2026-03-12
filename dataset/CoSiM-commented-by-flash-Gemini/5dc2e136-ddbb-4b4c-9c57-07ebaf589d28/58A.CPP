/// Problem Name: Chat room
/// Problem Link: https://codeforces.com/problemset/problem/58/A

/**
 * @5dc2e136-ddbb-4b4c-9c57-07ebaf589d28/58A.CPP
 * @brief This program solves Codeforces problem 58A, "Chat room".
 * It determines if the word "hello" can be formed as a subsequence
 * of a given input string 's'. A subsequence means the characters
 * 'h', 'e', 'l', 'l', 'o' must appear in 's' in that specific order,
 * but not necessarily consecutively.
 *
 * Algorithm: Two-pointer approach for subsequence matching.
 * Time Complexity: O(L) where L is the length of the input string 's'.
 * Space Complexity: O(1) for storing pointers and a fixed-size target string.
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes iostream, string.

using namespace std; // Uses the standard namespace.

int main() { // Corrected main function signature to return int.
  int i = 0, j = 0; // Initializes 'i' (pointer for input string 's') and 'j' (pointer for target string 'h').
  string s;         // Declares a string variable to store the input.
  string h = "hello"; // Defines the target string "hello".
  cin >> s;         // Reads the input string 's' from standard input.
  int len = s.size(); // Gets the length of the input string 's'.
  
  /**
   * Block Logic: Iterates through the input string 's' to find characters
   * that match the target string 'h' in sequence.
   * Invariant: 'j' tracks how many characters of "hello" have been found so far.
   */
  while (i < len && j < 5) { // Loop as long as both pointers are within valid bounds.
    // Inline: Checks if the character at the current position in 's' matches the current character in 'h'.
    if (s[i] == h[j]) { 
      i++; // Advances pointer for 's'.
      j++; // Advances pointer for 'h' (a match is found).
    } else {
      i++; // Only advances pointer for 's' (current character in 's' does not match, look further).
    }
  }
  
  /**
   * Block Logic: Determines the final output based on whether "hello" was found as a subsequence.
   * Pre-condition: The loop has completed, and 'j' indicates how many characters of "hello" were matched.
   * Invariant: Prints "YES" if all characters of "hello" were matched, otherwise "NO".
   */
  if (j == 5) // If 'j' reached 5, it means all characters of "hello" were found.
    puts("YES"); // Prints "YES".
  else
    puts("NO"); // Prints "NO".
  return 0; // Indicates successful program termination.
}
