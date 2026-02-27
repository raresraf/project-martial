/**
 * @file 58A.CPP
 * @brief Solution for competitive programming problem: "Chat Room".
 *
 * @details This program checks if the word "hello" is a subsequence of a given input string.
 *          A subsequence is formed by deleting zero or more characters from another string
 *          without changing the order of the remaining characters.
 *          Algorithm: Two-pointer approach for subsequence matching.
 *          Time Complexity: O(L), where L is the length of the input string, as each
 *                           character is visited at most once.
 *          Domain: Competitive Programming.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string s, t; // Declare strings: 's' for the input, 't' for the target subsequence "hello".
  cin >> s;    // Read the input string.
  t = "hello"; // Initialize the target subsequence.
  
  // Precondition: If the input string is shorter than "hello", it cannot contain "hello" as a subsequence.
  if (s.length() < 5) {
    cout << "NO" << endl; // Output "NO".
    return 0;             // Terminate program.
  }

  int i = 0, j = 0; // Initialize two pointers: 'i' for input string 's', 'j' for target string 't'.
  
  // Block Logic: Iterate through the input string to find characters of 't' in sequence.
  // Invariant: 'i' advances through 's', 'j' advances through 't' only on a match.
  while (i < s.length() && j < 5) { // Continue as long as both pointers are within bounds.
    // If the current character in 's' matches the current character in 't'.
    if (s[i] == t[j]) {
      i++; // Advance both pointers.
      j++;
    } else {
      i++; // If no match, only advance 'i' (move to next character in input string).
    }
  }
  
  // Block Logic: After the loop, check if all characters of "hello" were found.
  // Precondition: If 'j' has reached 5, it means all characters of "hello" were found in order.
  if (j >= 5) {
    cout << "YES" << endl; // Output "YES".
  } else {
    cout << "NO" << endl;  // Output "NO".
  }
}