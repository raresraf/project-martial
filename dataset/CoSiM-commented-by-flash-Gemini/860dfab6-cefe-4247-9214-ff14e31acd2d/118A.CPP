/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces (Problem 118A).
 * This program transforms an input string by converting uppercase letters to lowercase,
 * deleting all vowels ('a', 'o', 'y', 'e', 'u', 'i'), and inserting a "." before each consonant.
 *
 * Algorithm: Character-by-character processing. Each character is checked for being a vowel
 * or a consonant, converted to lowercase, and then appended to an output string with a
 * preceding dot if it's a consonant.
 * Time Complexity: O(L), where L is the length of the input string, as each character is processed once.
 * Space Complexity: O(L) for storing the input and output strings.
 */

#include <bits/stdc++.h> // @brief Includes many standard libraries (e.g., iostream, string, algorithm, vector, cctype).
#define LOG(x) cout << x << "\n" // @brief Macro for printing a value followed by a newline.
// -std=c++11 // @brief Compiler flag indicating C++11 standard usage.
using namespace std;

bool isVowel(char x) {
  /**
   * @brief Checks if a given character is a vowel (case-insensitive).
   * Vowels include 'a', 'e', 'i', 'o', 'u', and 'y'.
   * @param x The character to check.
   * @return True if the character is a vowel, false otherwise.
   */
  // Block Logic: Define a list of lowercase vowels.
  // The problem statement includes 'y' as a vowel.
  vector<char> vowels = {'a', 'e', 'i', 'o', 'u', 'y'};
  // Block Logic: Iterate through the vowels to check for a match.
  // It checks both the character itself and its uppercase equivalent (by subtracting 32 from ASCII).
  for (char c : vowels) {
    // 32 is diff b/w ASCII 'A' and 'a'
    if ((x == c) || (x == c - 32)) { // Inline: Case-insensitive comparison.
      return true;
    }
  }
  return false;
}

int main() {
  /**
   * @brief Main function to perform the string transformation task.
   * Reads an input string, iterates through its characters, and builds a new
   * string according to the transformation rules (lowercase consonants preceded by '.').
   */
  string s; // @brief The input string.
  cin >> s;
  int len = s.size(); // @brief Length of the input string.
  string out = ""; // @brief The output string after transformation.
  
  // Block Logic: Iterate through each character of the input string.
  // Pre-condition: 's' contains the input string.
  // Invariant: 'out' is built by appending transformed characters.
  for (int i = 0; i < len; i++) {
    // Block Logic: Apply transformation rules.
    // If the character is not a vowel.
    if (!isVowel(s[i])) {
      out.push_back('.'); // Inline: Insert "." before each consonant.
      out.push_back(tolower(s[i])); // Inline: Append the lowercase consonant.
    }
  }
  LOG(out); // @brief Print the transformed string followed by a newline.
  return 0; // @brief Indicates successful program termination.
}