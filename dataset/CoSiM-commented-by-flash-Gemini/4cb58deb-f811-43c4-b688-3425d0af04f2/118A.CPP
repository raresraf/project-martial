/**
 * @file 118A.CPP
 * @brief Processes an input word by converting to lowercase, removing vowels, and formatting consonants with preceding dots.
 *
 * Algorithm: Utilizes C++ STL functions (`std::transform`, `std::remove`, `std::erase`) for efficient
 *            case normalization, vowel filtering, and subsequent formatted output.
 * Time Complexity: O(L * V) where L is the length of the input string and V is the number of distinct vowels removed.
 *                  Each removal operation (remove + erase) can take O(L).
 * Space Complexity: O(L) for storing and manipulating the input string.
 */
// Written By Deepesh Nimma
// Codeforces: 118A Codeforces
#include <bits/stdc++.h> // Includes most standard C++ libraries, especially <string>, <algorithm>, <iostream>.
using namespace std;

#define ll long long   // Functional Utility: Defines 'll' as a shorthand for 'long long'.
#define end1 "\n"      // Functional Utility: Defines 'end1' as a shorthand for newline character, often for faster I/O.

/**
 * @brief Main function to process the input word as per problem rules.
 * @pre Input consists of a single word (string).
 * @post Prints the processed string with lowercase consonants, each preceded by a dot.
 */
int main() {
  string s; // Variable to store the input word.
  // Functional Utility: Reads the input word into the string 's'.
  cin >> s;

  // Functional Utility: Converts all characters in the string 's' to lowercase.
  // Uses std::transform with ::tolower for case normalization.
  transform(s.begin(), s.end(), s.begin(), ::tolower);

  /**
   * Block Logic: Removes all occurrences of specified vowels from the string.
   * Invariant: Each call to `s.erase(remove(...))` removes all instances of a particular vowel
   *            by shifting non-vowel characters and resizing the string.
   */
  s.erase(remove(s.begin(), s.end(), 'a'), s.end()); // Removes all 'a's.
  s.erase(remove(s.begin(), s.end(), 'e'), s.end()); // Removes all 'e's.
  s.erase(remove(s.begin(), s.end(), 'i'), s.end()); // Removes all 'i's.
  s.erase(remove(s.begin(), s.end(), 'o'), s.end()); // Removes all 'o's.
  s.erase(remove(s.begin(), s.end(), 'u'), s.end()); // Removes all 'u's.
  s.erase(remove(s.begin(), s.end(), 'y'), s.end()); // Removes all 'y's.

  /**
   * Block Logic: Iterates through the modified string and prints each character preceded by a dot.
   * Invariant: Each character in the remaining string (which are now consonants) is printed with a '.'.
   */
  for (int i = 0; i < s.size(); ++i) {
    cout << '.' << s[i]; // Prints a dot followed by the character.
  }
  // No explicit return 0; is needed for main in C++ if it reaches the end.
  // However, for competitive programming, it's often good practice to include it.
}
