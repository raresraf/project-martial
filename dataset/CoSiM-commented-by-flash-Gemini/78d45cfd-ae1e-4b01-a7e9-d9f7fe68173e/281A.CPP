/**
 * @file 281A.CPP
 * @brief Solution to Codeforces problem 281A - Word Capitalization.
 *        This program takes a single word as input and capitalizes its first letter,
 *        leaving the rest of the word unchanged.
 * Algorithm:
 *   1. Read the input word as a string.
 *   2. Extract the first character of the string.
 *   3. Convert this first character to its uppercase equivalent.
 *   4. Remove the original first character from the input string.
 *   5. Concatenate the capitalized first character with the modified (rest of the) string.
 *   6. Print the resulting capitalized word.
 * Time Complexity: O(L), where L is the length of the input string, due to string manipulation operations.
 * Space Complexity: O(L) for storing the input string and potentially a temporary string for the first character.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * @brief Main function implementing the solution for the "Word Capitalization" problem.
   *
   * It performs fast I/O setup, reads a single word, capitalizes its first letter,
   * and prints the modified word to standard output.
   * @return int Returns 0 upon successful execution.
   */
  ios::sync_with_stdio(false);
  cin.tie(0); // Unties cin from cout for faster input.

  string s, s1;
  cin >> s; // Read the input word.

  // Block Logic: Extracts the first character and converts it to uppercase.
  // Pre-condition: 's' contains the input word.
  // Post-condition: 's1' contains the uppercase version of the first character of 's'.
  s1 = s.at(0);
  transform(s1.begin(), s1.end(), s1.begin(), ::toupper);
  
  // Block Logic: Removes the original first character from the input string.
  // Pre-condition: 's' contains the input word.
  // Post-condition: 's' now contains the original word starting from its second character.
  s.erase(0, 1);

  // Block Logic: Concatenates the capitalized first character with the rest of the string and prints.
  // Pre-condition: 's1' holds the capitalized first character, 's' holds the rest of the word.
  // Post-condition: The word with its first letter capitalized is printed.
  cout << s1 + s << "\n";
}
