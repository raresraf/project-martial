/**
 * @file 112A.CPP
 * @brief This program solves the "Petya and Strings" problem (Codeforces 112A).
 *   It compares two input strings lexicographically, ignoring case.
 *   Problem description: https://codeforces.com/problemset/problem/112/A
 * * Algorithm: Manual case-insensitive lexicographical comparison.
 *   1. Reads two input strings.
 *   2. Converts all uppercase characters in both strings to lowercase using ASCII arithmetic.
 *   3. Compares the characters of the two strings one by one until a difference is found or an end is reached.
 *   4. Prints -1 if the first string is smaller, 1 if the second is smaller, and 0 if they are equal.
 * Time Complexity: O(L) where L is the length of the strings (assuming they are of similar length),
 *   due to iterating through the strings once for lowercase conversion and once for comparison.
 * Space Complexity: O(L) to store the input strings.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, including <iostream> and <string>.

using namespace std;

/**
 * @brief Main function to perform case-insensitive string comparison.
 * Functional Utility: Reads two strings, converts them to lowercase manually,
 *                     compares them character by character to find lexicographical order,
 *                     and prints the comparison result (-1, 0, or 1).
 */
int main() {
  string s; // Stores the first input string.
  string p; // Stores the second input string.
  // Block Logic: Reads the two input strings, potentially including spaces, using `getline`.
  getline(cin, s);
  getline(cin, p);
  int len = s.size(); // Stores the length of the strings. Assumes both strings have the same length.
  /**
   * Block Logic: Converts any uppercase characters in both strings `s` and `p` to lowercase.
   * Pre-condition: Strings `s` and `p` contain characters from the English alphabet.
   * Invariant: After this loop, both strings `s` and `p` will contain only lowercase characters.
   * Inline: `s[i] >= 'A' && s[i] <= 'Z'` checks if the character is uppercase.
   * Inline: `s[i] = (s[i] - 'A') + 'a'` converts the character to lowercase using ASCII arithmetic.
   */
  for (int i = 0; i < len; i++) {
    if (s[i] >= 'A' && s[i] <= 'Z')
      s[i] = (s[i] - 'A') + 'a';
    if (p[i] >= 'A' && p[i] <= 'Z')
      p[i] = (p[i] - 'A') + 'a';
  }

  /*
   * Block Logic: This commented-out section appears to be an attempt at a bubble sort algorithm.
   * It is not relevant to the problem of lexicographical string comparison and is effectively dead code.
   *
  for(int i = 0; i < len - 1; i++){
      bool sw = false;
      for(int j = 0; j < len - 1 - i; j++){
          if(s[j] > s[j + 1]){
              int temp = s[j];
              s[j] = s[j + 1];
              s[j + 1] = temp;
              sw = true;
          }
      }
      if(sw == false) break;
  }

  for(int i = 0; i < len - 1; i++){
      bool sw = false;
      for(int j = 0; j < len - 1 - i; j++){
          if(p[j] > p[j + 1]){
              int temp = p[j];
              p[j] = p[j + 1];
              p[j + 1] = temp;
              sw = true;
          }
      }
      if(sw == false) break;
  }*/
  // `c1`: Flag set if `s` is found to be lexicographically greater than `p`.
  // `c2`: Flag set if `s` is found to be lexicographically smaller than `p`.
  int c1 = 0, c2 = 0;

  /**
   * Block Logic: Performs character-by-character lexicographical comparison.
   * It iterates until a difference is found or the end of the strings is reached.
   * Pre-condition: Both strings `s` and `p` are of the same length and in lowercase.
   * Invariant: `c1` or `c2` will be set upon the first differing character, leading to an early break.
   */
  for (int i = 0; i < len; i++) {
    if (s[i] > p[i]) {
      c1++;  // `s` is greater than `p`.
      break; // Early exit as the lexicographical order is determined.
    } else if (s[i] < p[i]) {
      c2++;  // `s` is smaller than `p`.
      break; // Early exit as the lexicographical order is determined.
    }
  }
  /**
   * Conditional Logic: Prints the comparison result based on `c1` and `c2` flags.
   * - If `c1 > c2` (meaning `c1` was set and `c2` was not), `s` is lexicographically greater.
   * - If `c1 < c2` (meaning `c2` was set and `c1` was not), `s` is lexicographically smaller.
   * - If `c1 == c2` (meaning neither was set, implies `0` for both), strings are equal.
   */
  if (c1 > c2)
    printf("1\n");
  else if (c1 < c2)
    printf("-1\n");
  else if (c1 == c2)
    printf("0\n");

  return 0;
}
