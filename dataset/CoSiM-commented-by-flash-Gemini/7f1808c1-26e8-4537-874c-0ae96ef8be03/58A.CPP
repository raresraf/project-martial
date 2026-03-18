/**
 * @file 58A.CPP
 * @brief This program determines if the word "hello" can be formed as a subsequence of a given input string.
 * Algorithm: Two-pointer approach (or greedy approach) to check for subsequence existence.
 * Time Complexity: O(L) where L is the length of the input string.
 * Space Complexity: O(1) as only a few variables are used to store state.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to execute the "hello" subsequence detection.
 * It reads an input string and checks if "hello" can be formed by deleting zero or more characters from it.
 */
int main() {
  string s, r = "hello";
  int i, j = 0;
  cin >> s;
  /**
   * @brief Iterates through the input string 's' to find characters matching the "hello" subsequence 'r'.
   * Pre-condition: 's' is the input string, 'r' is the target subsequence "hello".
   * 'j' tracks the current matched character index in 'r'.
   * Invariant: 'j' only increments when a character from 's' matches the character at 'r[j]',
   * effectively checking for the subsequence in order.
   */
  for (int i = 0; i < s.length(); i++) {
    // If the current character in 's' matches the next expected character in 'r', advance 'r's index.
    if (s[i] == r[j])
      j++;
  }
  /**
   * @brief Determines and prints whether "hello" was found as a subsequence.
   * Pre-condition: After iterating through 's', 'j' contains the count of matched characters from 'r'.
   * Post-condition: Prints "YES" if all characters of "hello" were found in order, otherwise "NO".
   */
  if (j == 5)
    printf("YES");
  else
    printf("NO");
}
