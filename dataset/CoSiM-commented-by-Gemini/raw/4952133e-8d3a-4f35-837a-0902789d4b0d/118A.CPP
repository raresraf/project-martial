/**
 * @file 118A.CPP
 * @brief Solution for the "String Task" programming contest problem.
 * @details This program processes a string according to three rules:
 *          1. Delete all vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 *          2. Insert a '.' character before each consonant.
 *          3. Convert all uppercase consonants to their lowercase equivalents.
 *          This implementation builds the resulting string before printing.
 *
 * Algorithm:
 * 1. Read the input string `s` and initialize an empty result string `r`.
 * 2. Iterate through each character of the input string `s`.
 * 3. For each character:
 *    a. Convert it to its lowercase equivalent. This simplifies the vowel check.
 *    b. Check if the lowercase character is a vowel.
 *    c. If it is NOT a vowel (i.e., it's a consonant):
 *       i. Append a '.' character to the result string `r`.
 *       ii. Append the lowercase consonant itself to `r`.
 * 4. After iterating through all characters, print the final result string `r`.
 *
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(L) to store both the input and result strings.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s, r;
  cin >> s;
  
  /**
   * @block
   * @description This loop processes each character of the input string `s`
   * and builds the transformed string `r`.
   */
  for (int i = 0; i < s.size(); i++) {
    char c = s[i];
    
    // First, convert the character to lowercase.
    if (c >= 'A' && c <= 'Z') {
      c += 'a' - 'A'; // Portable way to convert to lowercase.
      // c+= 32;
    }
    
    // Process the character if it is a consonant.
    if (c != 'a' && c != 'e' && c != 'i' && c != 'o' && c != 'u' && c != 'y') {
      // Prepend a '.' to the consonant.
      r.append(1, '.');
      // Append the lowercase consonant.
      r.append(1, c);
    }
  }
  
  // Print the fully constructed result string.
  cout << r << endl;
  return 0;
}