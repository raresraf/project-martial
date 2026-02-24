/**
 * @file 236A.CPP
 * @brief Solution for Codeforces Problem 236A - Boy or Girl.
 *
 * This program determines if a given username should be categorized as
 * "CHAT WITH HER!" or "IGNORE HIM!" based on the parity of the number of
 * distinct characters in the username.
 *
 * Algorithm: Sorting and counting distinct characters. The string is sorted
 * to group identical characters, then a linear scan counts the unique characters.
 * The parity of this count determines the output.
 * Time Complexity: O(N log N) due to sorting, where N is the length of the string.
 * Space Complexity: O(N) for storing the string.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * @brief Main function to execute the Boy or Girl problem logic.
   *
   * Reads a username string, sorts it, counts the number of distinct characters,
   * and prints a message based on whether this count is even or odd.
   */
  // Declare a character array to store the input string (username), with a maximum size of 100.
  char s[100];
  // Read the input username string.
  cin >> s;
  // Declare integer variables for loop counters and 'c' to count distinct characters.
  int i, j, c = 0;
  // Calculate the length of the input string.
  int l = strlen(s);
  // Sort the characters in the string. This groups identical characters together,
  // making it easier to count distinct ones.
  sort(s, s + l);

  // Block Logic: Iterate through the sorted string to count distinct characters.
  // Invariant: 'i' traverses the string, and 'c' increments only when a character
  // is different from its successor, indicating a new distinct character.
  for (i = 0; i < l; i++) {
    // If the current character is not the same as the next character, it's a distinct character.
    if (s[i] != s[i + 1]) {
      c++; // Increment the distinct character count.
    }
  }
  // Block Logic: Determine the output based on the parity of distinct character count.
  // Pre-condition: 'c' holds the total number of distinct characters in the username.
  if (c % 2 == 0) {
    // If the count of distinct characters is even, print "CHAT WITH HER!".
    cout << "CHAT WITH HER!";
  } else
    // If the count of distinct characters is odd, print "IGNORE HIM!".
    cout << "IGNORE HIM!";
  return 0;
}
