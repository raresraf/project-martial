/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from competitive programming.
 *
 * @details
 * This program manipulates a string according to the following rules:
 * 1. Deletes all vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 * 2. Inserts a '.' before each consonant.
 * 3. Converts all consonants to lowercase.
 *
 * This implementation reads the entire string into memory, builds a new output
 * string, and then prints the result.
 *
 * Time Complexity: O(N), where N is the length of the input string. The isVowel function takes constant time (as the vector is of fixed size 6), so the main loop dominates.
 * Space Complexity: O(N) to store both the input and output strings.
 */
#include <bits/stdc++.h>
#define LOG(x) cout << x << "\n" // Macro for printing with a newline.
// -std=c++11
using namespace std;

/**
 * @brief Checks if a given character is a vowel (case-insensitive).
 * Vowels are defined as 'a', 'e', 'i', 'o', 'u', 'y'.
 * @param x The character to check.
 * @return True if the character is a vowel, false otherwise.
 */
bool isVowel(char x) {
  // The problem statement includes 'y' as a vowel
  vector<char> vowels = {'a', 'e', 'i', 'o', 'u', 'y'};
  for (char c : vowels) {
    // Check for both lowercase and uppercase by comparing with the ASCII difference.
    // 32 is the difference between ASCII 'a' and 'A'.
    if ((x == c) || (x == c - 32)) {
      return true;
    }
  }
  return false;
}

int main() {
  string s;
  cin >> s; // Read the entire input string.
  int len = s.size();
  string out = ""; // Initialize an empty string to build the output.
  /**
   * @brief Iterates through the input string to process each character.
   */
  for (int i = 0; i < len; i++) {
    // Process the character only if it is a consonant.
    if (!isVowel(s[i])) {
      // Insert "." before each consonant.
      out.push_back('.');
      // Substitute uppercase consonants with lowercase and append to output.
      out.push_back(tolower(s[i]));
    }
  }
  LOG(out); // Print the final transformed string.
  return 0;
}