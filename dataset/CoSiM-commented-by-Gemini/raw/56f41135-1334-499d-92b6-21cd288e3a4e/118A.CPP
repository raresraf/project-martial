/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces.
 * @details The program processes a string according to three rules:
 * 1. Delete all vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 * 2. Insert a '.' before each consonant.
 * 3. Convert all uppercase consonants to their lowercase equivalents.
 * This implementation processes the string character by character and prints the
 * output on the fly, without building an intermediate result string.
 *
 * @algorithm
 * 1. Read the input string.
 * 2. Loop through each character of the string.
 * 3. In each iteration:
 *    a. First, convert the character to lowercase. This implementation does
 *       this by checking its ASCII value and adding 32 if it's uppercase.
 *    b. Check if the now-lowercase character is a vowel.
 *    c. If it is a vowel, use `continue` to skip to the next character.
 *    d. If it is not a vowel (i.e., a consonant), print a '.' followed by the
 *       lowercase consonant character.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the string, due to a single pass.
 * Space Complexity: O(L) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n; // Declared but unused.
  string s;
  cin >> s;
  
  /**
   * @block
   * @brief Process each character of the string one by one.
   */
  for (int i = 0; i < s.length(); i++) {
    // Convert character to lowercase using ASCII arithmetic.
    // This checks if the char is uppercase (ASCII <= 90 for 'Z').
    if (s[i] <= 90)
      s[i] += 32;
    
    // Check if the character is a vowel.
    if (s[i] == 'a' || s[i] == 'o' || s[i] == 'y' || s[i] == 'e' ||
        s[i] == 'u' || s[i] == 'i') {
      // If it's a vowel, skip it.
      continue;
    } else {
      // If it's a consonant, print a '.' followed by the character.
      cout << "." << s[i];
    }
  }
}
