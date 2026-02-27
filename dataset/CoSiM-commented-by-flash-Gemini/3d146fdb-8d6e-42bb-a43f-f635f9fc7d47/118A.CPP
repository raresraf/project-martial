/**
 * @file 118A.CPP
 * @brief Solution for competitive programming problem: "String Task".
 *
 * @details This program processes an input string by removing all vowels
 *          (A, O, Y, E, U, I, and their lowercase counterparts). For each
 *          remaining consonant, it prepends a '.' and converts the consonant to lowercase,
 *          building a new string with the transformed characters.
 *          Algorithm: Iterative string traversal and character-by-character building of a new string.
 *          Time Complexity: O(L), where L is the length of the input string, as each
 *                           character is processed a constant number of times.
 *          Domain: Competitive Programming.
 */
#include <bits/stdc++.h>
#include <iostream>
#include <string>
using namespace std;
int main() {
  // Declare strings: 's' to store the input, 'n' to build the transformed output.
  string s, n;
  n = ""; // Initialize the output string as empty.
  // Read the input string.
  cin >> s;
  
  // The commented-out line is part of the original code, possibly for debugging.
  // cout<<s<<endl;

  // Block Logic: Iterate through each character of the input string to perform transformations.
  // Invariant: 's' is the original input string; 'n' is built as the transformed output.
  for (int i = 0; i < s.length(); i++) {
    // Block Logic: Check if the current character is a vowel (case-insensitive).
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y' || s[i] == 'A' || s[i] == 'E' ||
        s[i] == 'I' || s[i] == 'O' || s[i] == 'U' || s[i] == 'Y') {
      continue; // If it's a vowel, skip to the next character (effectively removing it).
    }

    n = n + '.'; // Prepend a dot to the output string.
    // Block Logic: Convert uppercase consonants to lowercase.
    if (s[i] >= 'A' && s[i] <= 'Z')
      s[i] = s[i] + 32; // ASCII conversion from uppercase to lowercase.
    n = n + s[i]; // Append the (possibly converted to lowercase) consonant to the output string.
  }

  // Output the final transformed string.
  cout << n << endl;
  
  return 0; // Indicate successful program termination.
}
