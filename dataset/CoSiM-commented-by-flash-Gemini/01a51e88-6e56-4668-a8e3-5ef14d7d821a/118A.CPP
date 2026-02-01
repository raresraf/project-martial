/**
 * @file 118A.CPP
 * @brief Solution for a competitive programming problem (Codeforces 118A)
 *        that involves processing a given string. The task is to convert
 *        all characters to lowercase, remove all vowels ('a', 'o', 'y', 'e', 'u', 'i'),
 *        and then insert a '.' before each remaining consonant.
 *
 * Algorithm: String manipulation with character-by-character processing.
 * Time Complexity: $O(L)$ where L is the length of the input string, due to linear scan and transformations.
 * Space Complexity: $O(L)$ to store the modified string, or $O(1)$ if output is streamed.
 */

#include <bits/stdc++.h> // Includes common C++ standard library headers.
using namespace std;

/**
 * @brief Main function to process the input string according to problem rules.
 * Functional Utility: Orchestrates the string transformation process, converting
 *                     to lowercase, filtering vowels, and prepending '.' to consonants.
 * @return 0 on successful execution.
 */
int main() {
/**
 * @brief Block Logic: Conditional compilation for local file I/O redirection.
 * Functional Utility: This block is typically used in competitive programming
 *                     to redirect standard input/output to files during local
 *                     testing, allowing for easier debugging with sample cases.
 * Invariant: This block is only active when `ONLINE_JUDGE` is not defined (i.e., local environment).
 */
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin); // Redirects standard input from "input.txt".
  freopen("output.txt", "w", stdout); // Redirects standard output to "output.txt".
#endif
  string s, s1; // Declares two strings: `s` for input, `s1` for modified output.
  cin >> s; // Reads the input string.
  int l = s.length(); // Stores the length of the input string.
  // Transforms all characters in 's' to lowercase.
  transform(s.begin(), s.end(), s.begin(), ::tolower);
  /**
   * @brief Block Logic: Iterates through the lowercase string to filter vowels and format consonants.
   * Invariant: For each character `s[i]`, it is checked if it is a vowel or a consonant.
   * Precondition: `s` contains the lowercase version of the input string.
   * Postcondition: `s1` is populated with consonants preceded by a '.', and vowels are omitted.
   */
  for (int i = 0; i < l; i++) {
    /**
     * @brief Block Logic: Checks if the current character is a vowel.
     * Precondition: `s[i]` is a lowercase character.
     * Postcondition: If `s[i]` is a vowel, it is skipped; otherwise, it is processed as a consonant.
     */
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y')
      continue; // Skips the character if it is a vowel.
    else {
      s1.push_back('.'); // Appends a '.' to the result string.
      s1.push_back(s[i]); // Appends the consonant to the result string.
    }
  }
  cout << s1 << "\n"; // Prints the final modified string followed by a newline.
  return 0; // Indicates successful program termination.
}