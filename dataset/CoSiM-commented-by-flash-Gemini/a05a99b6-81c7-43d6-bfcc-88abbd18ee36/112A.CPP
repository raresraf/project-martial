/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem from Codeforces (Problem 112A).
 *        This program compares two input strings lexicographically, ignoring case.
 *        It converts both strings to lowercase and then determines their relative
 *        order, printing 0 if they are equal, -1 if the first is smaller,
 *        and 1 if the first is larger.
 * Algorithm: String Comparison (Lexicographical)
 * Time Complexity: O(L), where L is the length of the strings (assuming string lengths are equal).
 *                  Each character conversion and comparison takes constant time.
 * Space Complexity: O(L) to store the input strings.
 */

#include <bits/stdc++.h> // Includes most standard C++ libraries, including iostream and string.
using namespace std;

/**
 * @brief Main function to read, normalize, and compare two strings lexicographically.
 * @return 0 if the program executes successfully.
 */
int main() {
  string s1, s2; // Declare two strings to store the input.
  
  /**
   * Block Logic: Continuously reads pairs of strings, converts them to lowercase, and compares them.
   * Invariant: The loop continues as long as input pairs are provided.
   */
  while (cin >> s1 >> s2) { // Read two strings from standard input.
    /**
     * Block Logic: Converts both strings to lowercase for case-insensitive comparison.
     * Invariant: After each iteration, s1[i] and s2[i] are converted to their lowercase equivalents.
     */
    for (int i = 0; i < s1.size(); i++) { // Iterate through the characters of the first string (assuming equal length).
      // Convert the character at the current index in s1 to lowercase.
      s1[i] = tolower(s1[i]);
      // Convert the character at the current index in s2 to lowercase.
      s2[i] = tolower(s2[i]);
    }
    
    /**
     * Block Logic: Compares the normalized strings lexicographically and prints the result.
     * Pre-condition: s1 and s2 contain lowercase versions of the input strings.
     * Post-condition: "0", "-1", or "1" is printed based on the comparison.
     */
    if (s1 == s2) // If the strings are equal after normalization.
      cout << "0" << endl; // Print 0.
    else if (s1 < s2) // If the first string is lexicographically smaller.
      cout << "-1" << endl; // Print -1.
    else // If the first string is lexicographically larger.
      cout << "1" << endl; // Print 1.
  }
  return 0; // Indicate successful program termination.
}
