/**
 * @file 118A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 118A - "String Task").
 *
 * This program transforms an input string according to specific rules:
 * 1. All uppercase letters are converted to their lowercase counterparts.
 * 2. All vowels (a, o, y, e, u, i, case-insensitive) are deleted.
 * 3. A "." character is inserted before each remaining consonant.
 *
 * Algorithm: String processing with character-by-character transformation.
 * It iterates through the input string, converts each character to lowercase.
 * If the character is a vowel, it is skipped. If it is a consonant, a "." is
 * printed followed by the consonant.
 *
 * Time Complexity: O(L), where L is the length of the input string `a`.
 * This involves a single pass over the input string.
 * Space Complexity: O(1) (excluding input string storage), as no additional
 * data structures are dynamically grown based on input size.
 */
#include <ctype.h>  // Required for tolower() function.
#include <iostream> // Required for input/output operations (cin, cout).
using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads an input string, applies the transformation rules (lowercase, remove vowels, prefix consonants),
 * and prints the modified string to standard output.
 */
int main() {
  string a;       // Variable to store the input string.
  int i, j, n;    // Loop counter 'i', 'j' unused, 'n' stores string length.
  // Pre-condition: Read the input string.
  cin >> a;
  n = a.size(); // Functional Utility: Get the length of the string.
  // Block Logic: Iterate through each character of the input string.
  // Invariant: 'i' represents the current index in the string.
  for (i = 0; i < n; i++) {
    a[i] = tolower(a[i]); // Functional Utility: Convert the current character to lowercase.
    // Block Logic: Check if the character is a vowel.
    if (a[i] == 'a' || a[i] == 'e' || a[i] == 'i' || a[i] == 'o' ||
        a[i] == 'u' || a[i] == 'y')
      continue; // If it's a vowel, skip to the next character.
    else        // If it's a consonant.
      cout << "." << a[i]; // Print a dot followed by the consonant.
  }
  cout << endl; // Add a newline at the end for cleaner output.
  return 0;     // Indicate successful execution.
}
