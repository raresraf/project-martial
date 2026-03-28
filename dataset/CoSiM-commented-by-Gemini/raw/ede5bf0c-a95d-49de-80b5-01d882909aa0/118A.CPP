/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from competitive programming platforms.
 *
 * The task is to process a string according to the following rules:
 * 1. Delete all vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 * 2. Insert a character "." before each consonant.
 * 3. Replace all uppercase consonants with their lowercase equivalents.
 *
 * @b Algorithm: The program first converts the entire string to lowercase. Then, it
 *    iterates through the modified string. For each character, it checks if it is
 *    a vowel. If not (i.e., it's a consonant), it prints a "." followed by the
 *    character.
 * @b Time_Complexity: O(L), where L is the length of the string, due to the
 *    multiple passes over the string.
 * @b Space_Complexity: O(L) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

  string s;
  cin >> s;

  // First, convert the entire string to lowercase.
  for (int i = 0; i < s.length(); i++) {
    s[i] = tolower(s[i]);
  }

  /**
   * @brief Iterate through the string to process each character.
   */
  for (int i = 0; i < s.length(); i++) {
    // Check if the character is a vowel.
    // Note: The check for uppercase vowels is redundant since the string
    // has already been converted to lowercase.
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y')
      continue; // Skip vowels.
    else {
      // If the character is a consonant, print a '.' before it.
      cout << "." << s[i];
    }
  }
  cout << endl;

  return 0;
}