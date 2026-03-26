/**
 * @file 118A.CPP
 * @brief Solution to Codeforces problem 118A - String Task.
 * @description This program processes a string according to a specific set of rules:
 *              1. Deletes all vowels (A, O, Y, E, U, I).
 *              2. Inserts a character "." before each consonant.
 *              3. Replaces all uppercase consonants with their corresponding lowercase ones.
 *
 * Algorithm: The program iterates through the input string character by character. For each
 * character, it first converts it to lowercase. Then, it checks if the character is a
 * consonant (i.e., not one of 'a', 'o', 'y', 'e', 'u', 'i'). If it is a consonant, the
 * program prints a '.' followed by the lowercase consonant. This is done in a single pass.
 *
 * Time Complexity: O(N), where N is the length of the string, as it involves a single loop
 *                  through the string.
 * Space Complexity: O(N) to store the input string. The output is streamed, so additional
 *                   space for the output string is not explicitly allocated.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  /**
   * Block Logic: Iterate through each character of the string to apply the transformation rules.
   */
  for (int i = 0; i < s.size(); i++) {
    // Rule 3: Convert the character to lowercase.
    s[i] = tolower(s[i]);
    
    // Rule 1: Check if the character is a consonant (not a vowel).
    // 'y' is treated as a vowel according to the problem statement.
    if (s[i] != 'a' && s[i] != 'e' && s[i] != 'i' && s[i] != 'o' &&
        s[i] != 'u' && s[i] != 'y')
      // Rule 2: If it's a consonant, print a '.' before it.
      cout << "." << s[i];
  }
  cout << endl;
  return 0;
}
