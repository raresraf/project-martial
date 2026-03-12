/*
Link: https://codeforces.com/problemset/problem/112/A
@uthor: chavodon
*/
/**
 * @5cca10f9-c4af-4a10-8135-ece7fcf139c8/112A.CPP
 * @brief This program solves problem 112A ("Petya and Strings").
 * It performs a case-insensitive lexicographical comparison of two input strings.
 * It converts characters to lowercase during the comparison process and outputs
 * -1 if the first string is lexicographically smaller, 1 if it is larger,
 * and 0 if both strings are equal.
 *
 * Algorithm: Character-by-character comparison with on-the-fly lowercase conversion.
 * Time Complexity: O(L) where L is the length of the strings (assuming equal length).
 * Space Complexity: O(L) for storing the strings.
 */

#include <iostream> // For standard input/output operations (cin, cout).
#include <string>   // For using std::string.
#include <cctype>   // For std::tolower.

using namespace std; // Uses the standard namespace.

int main() {
  string a, b; // Declares two string variables to store the input words.
  int i, flag = 0; // Declares 'i' for the loop counter and 'flag' to indicate if a difference was found.
  cin >> a; // Reads the first string from standard input.
  cin >> b; // Reads the second string from standard input.
  
  /**
   * Block Logic: Iterates through the strings, comparing characters lexicographically after converting them to lowercase.
   * Invariant: At each step, characters up to 'i-1' have been found to be equal.
   *            'flag' is set to 1 and loop breaks if a difference is found.
   */
  for (i = 0; i < a.size(); i++) { // Loops through the characters of the strings.
                                    // Assumes both strings have at least this length, or comparison is only needed up to a.size().
    a[i] = tolower(a[i]); // Inline: Converts character a[i] to lowercase.
    b[i] = tolower(b[i]); // Inline: Converts character b[i] to lowercase.
    
    /**
     * Block Logic: Compares the lowercase characters and determines the lexicographical order.
     * Invariant: If a difference is found, the program outputs the result and terminates the loop.
     */
    if (a[i] < b[i]) { // Inline: If a[i] is lexicographically smaller than b[i].
      cout << "-1";    // Output -1.
      flag = 1;        // Set flag to indicate a difference was found.
      break;           // Terminate the loop.
    } else if (a[i] > b[i]) { // Inline: If a[i] is lexicographically greater than b[i].
      cout << "1";     // Output 1.
      flag = 1;        // Set flag to indicate a difference was found.
      break;           // Terminate the loop.
    }
  }
  // Block Logic: If the loop completed without 'flag' being set to 1, it means all characters were equal.
  if (flag == 0) // If no difference was found during the character-by-character comparison.
    cout << "0"; // Output 0.
  // The program implicitly prints a newline character at the end of output.
  return 0; // Indicates successful program termination.
}