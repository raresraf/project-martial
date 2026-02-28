/**
 * @file 118A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 118A - "String Task").
 *
 * This program processes an input string according to specific rules:
 * 1. All uppercase letters are converted to their lowercase counterparts.
 * 2. All vowels (A, O, Y, E, U, I, case-insensitive) are deleted.
 * 3. A "." character is inserted before each remaining consonant.
 *
 * Algorithm: String processing with character-by-character transformation.
 * The solution iterates through the input string, converts each character to lowercase,
 * checks if it's a consonant (not a vowel), and appends it to the result string
 * with a preceding dot if it is a consonant.
 *
 * Time Complexity: O(L), where L is the length of the input string.
 * This involves a single pass over the input string.
 * Space Complexity: O(L) to store the resulting transformed string.
 */
#include <bits/stdc++.h> // Includes various standard libraries for C++.
using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads an input string, applies the transformation rules, and prints the modified string.
 */
int main() {
  string task; // Variable to store the input string.
  // Pre-condition: Read the input string.
  cin >> task;
  
  string result = ""; // Variable to build the modified string.
  // Define a set of lowercase vowels for efficient lookup.
  set<char> vowels = {'a', 'e', 'i', 'o', 'u', 'y'};
  
  // Block Logic: Iterate through each character of the input string.
  // Invariant: 's' represents the current character being processed.
  for (auto s : task) {
    s = tolower(s); // Convert the current character to lowercase.
    // Block Logic: Check if the character is a consonant (not a vowel and within 'b' to 'z').
    // Invariant: 's' is a lowercase letter.
    // Functional Utility: The condition `s >= 'b' && s <= 'z'` filters out non-alphabetic characters
    // and 'a', while `vowels.find(s) == vowels.end()` ensures it's not a vowel.
    if (s >= 'b' && s <= 'z' && vowels.find(s) == vowels.end()) {
      result += '.'; // Prepend a dot.
      result += s;   // Append the consonant.
    }
  }
  // Output the final transformed string followed by a newline.
  cout << result << "\n";
  return 0;
}