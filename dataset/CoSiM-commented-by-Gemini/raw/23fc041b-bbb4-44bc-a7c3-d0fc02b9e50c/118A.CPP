/**
 * @file 118A.CPP
 * @brief Solution for the "String Task" programming problem.
 * @details This program processes a string of Latin letters based on a specific set of rules.
 * It first converts the entire string to lowercase. Then, it iterates through the string,
 * deleting all vowels ('a', 'o', 'y', 'e', 'u', 'i'), and for every consonant,
 * it prepends a '.' character. The final transformed string is then printed.
 *
 * @algorithm Single-pass string transformation.
 * @complexity
 * Time Complexity: O(L), where L is the length of the input string. This is because the program
 * performs a constant number of passes over the string (one for converting to lowercase, and
 * one for building the result).
 * Space Complexity: O(L), to store the input string and the resulting output string, which can
 * be up to twice the length of the input.
 */
#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  cin >> s;

  // Pre-processing step: Convert the entire string to lowercase to simplify vowel/consonant checks
  // and satisfy the requirement of converting uppercase consonants to lowercase.
  transform(s.begin(), s.end(), s.begin(), ::tolower);

  string t = "";
  // Block Logic: Iterate through each character of the lowercase string to build the result.
  for (int i = 0; i < s.length(); i++) {
    // Block Invariant: Check if the character is a vowel. The problem defines vowels as
    // "A", "O", "Y", "E", "U", "I".
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y')
      // If it's a vowel, skip it, effectively deleting it from the output.
      continue;
    else
      // If it's a consonant, prepend a '.' and append it to the result string.
      t = t + "." + s[i];
  }
  
  // Print the final transformed string.
  cout << t << endl;
  return 0;
}