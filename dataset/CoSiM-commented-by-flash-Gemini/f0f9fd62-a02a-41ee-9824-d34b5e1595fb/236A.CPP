/**
 * @file 236A.CPP
 * @brief This program solves the "Boy or Girl" problem (Codeforces 236A).
 *   It determines whether a given username (string) corresponds to a "masculine" or "feminine" persona
 *   based on the parity of the number of distinct characters it contains.
 *   If the number of distinct characters is even, it's "feminine" ("CHAT WITH HER!").
 *   If odd, it's "masculine" ("IGNORE HIM!").
 *   Problem description: https://codeforces.com/problemset/problem/236/A
 * * Algorithm: Frequency counting and parity check for distinct characters.
 *   It counts the frequency of each lowercase English letter present in the input string.
 *   Then, it counts the number of lowercase English letters that are *not* present in the string.
 *   The parity of this count is used to determine the output.
 * Time Complexity: O(L + 26) = O(L) where L is the length of the input string.
 * Space Complexity: O(26) = O(1) for the frequency array.
 */
#include <iostream> // Required for standard input/output operations.
#include <string>   // Required for std::string.
#include <vector>   // Included by <bits/stdc++.h>, but often used if not.

using namespace std;

/**
 * @brief Main function to determine the persona based on distinct characters.
 * Functional Utility: Reads a username, counts its distinct characters by first
 *                     counting missing characters, and outputs a message
 *                     ("CHAT WITH HER!" or "IGNORE HIM!") based on the count's parity.
 */
int main() {
  string str; // Stores the input username string.
  // Block Logic: Reads the input username string.
  cin >> str;
  // `a`: An integer array of size 26 to store the frequency of each lowercase English letter ('a' through 'z').
  // Inline: Initialized to all zeros.
  int a[26] = {0};
  // `ctr`: Counter for the number of lowercase English letters that are *not* present in the input string.
  int ctr = 0;
  /**
   * Block Logic: Populates the frequency array `a` by iterating through the input string.
   * Inline: `str[i] - 'a'` converts a character (e.g., 'b') to its 0-indexed position (e.g., 1).
   * Invariant: After this loop, `a[j]` contains the number of times the j-th letter of the alphabet appears in `str`.
   */
  for (int i = 0; i < str.length(); i++)
    a[str[i] - 'a']++;
  /**
   * Block Logic: Iterates through the frequency array to count characters that were *not* present.
   * Invariant: `ctr` accumulates the count of lowercase English letters that do not appear in `str`.
   */
  for (int i = 0; i < 26; i++) {
    if (a[i] < 1) // If the count for a character is less than 1, it means that character is not present.
      ++ctr;      // Increment the counter for missing distinct characters.
  }
  // Block Logic: Determines the output based on the parity of `ctr`.
  // If `ctr` (count of missing distinct characters) is even, it means `26 - ctr` (count of present distinct characters)
  // is also even. If `ctr` is odd, then `26 - ctr` is also odd.
  // The logic is: if `ctr` is even -> "CHAT WITH HER!", else -> "IGNORE HIM!".
  cout << (ctr % 2 == 0 ? "CHAT WITH HER!" : "IGNORE HIM!");
  return 0;
}