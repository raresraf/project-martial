/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" problem.
 *
 * This program transforms an input string according to specific rules:
 * 1. All vowels (a, o, y, e, u, i, case-insensitive) are removed.
 * 2. A '.' character is inserted before each consonant.
 * 3. All resulting characters are converted to lowercase.
 *
 * Algorithm: Character-by-character string processing with intermediate consonant string construction.
 * Time Complexity: O(L) where L is the length of the input string, due to single pass iterations.
 * Space Complexity: O(L) for the intermediate string 'a' to store consonants.
 */
#include <bits/stdc++.h>
using namespace std;
/**
 * @brief Main function to perform the string transformation.
 *
 * Functional Utility: Reads the input string, filters characters into an intermediate
 * consonant string, and then prints the transformed string to standard output.
 */
int main() {
  string s, a = ""; // 's' stores the input string, 'a' stores filtered consonants.
  cin >> s; // Read the input string.
  /**
   * Block Logic: Iterates through each character of the input string 's'.
   * Converts uppercase characters to lowercase.
   * If the character is a consonant (not a, e, i, o, u, y), it's appended to string 'a'.
   * Invariant: String 'a' accumulates all lowercase consonants from 's'.
   */
  for (int i = 0; i < s.size(); i++) {
    if (s[i] >= 'A' && s[i] <= 'Z') // Convert to lowercase if uppercase.
      s[i] = s[i] - 'A' + 'a';
    // Check if the character is a consonant.
    if (s[i] != 'a' && s[i] != 'e' && s[i] != 'i' && s[i] != 'o' &&
        s[i] != 'u' && s[i] != 'y')
      a += s[i]; // Append consonant to 'a'.
  }
  /**
   * Block Logic: Iterates through the string 'a' (containing only consonants).
   * Prints a '.' followed by each consonant.
   * Pre-condition: String 'a' contains all filtered lowercase consonants.
   * Post-condition: The transformed string (with '.' before each consonant) is printed.
   */
  for (int i = 0; i < a.size(); i++)
    cout << "." << a[i];
  printf("\n"); // Print a newline character at the end.
  return 0;
}
