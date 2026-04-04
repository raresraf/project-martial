/**
 * @file 118A.CPP
 * @brief Processes an input string by converting it to a specific "short form".
 *
 * This program reads a string, converts all its characters to lowercase, and then
 * constructs a new string by taking only the consonants. Each consonant is prefixed
 * with a dot ('.'). This is a common string manipulation problem.
 *
 * Algorithm: Iterative character processing. Uses a `std::set` for efficient vowel lookup.
 * Time Complexity: O(L), where L is the length of the input string, due to single pass
 *                  and O(log V) for set lookup (V is number of vowels, constant).
 *                  Effectively linear.
 * Space Complexity: O(1) for the vowel set and output buffering (excluding input string storage).
 */
#include <bits/stdc++.h> // Includes most standard library headers.
using namespace std;     // Uses the standard namespace.

typedef long long ll;     // Functional Utility: Defines 'll' as a shorthand for 'long long'.
typedef pair<int, int> ii; // Functional Utility: Defines 'ii' as a shorthand for 'pair<int, int>'.

int main() {
  /**
   * @brief Main function: Processes an input string, filtering consonants and formatting output.
   * @return 0 if the program executes successfully.
   */
  // Functional Utility: Optimizes C++ standard streams for faster input/output operations.
  // 'cin.tie(0)' prevents 'cin' from flushing 'cout' before each input operation.
  // 'ios_base::sync_with_stdio(0)' unties C++ streams from C standard streams.
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);
  
  // Functional Utility: A set to store lowercase vowels for efficient lookup.
  // 'y' is considered a vowel in this problem context.
  set<char> vwl = {'a', 'e', 'i', 'o', 'u', 'y'};
  string s; // Declare a string to hold the input.
  cin >> s; // Read the input string.

  /**
   * @brief Loop through each character of the input string to filter and format output.
   * @pre 's' contains the input string; 'vwl' contains the set of vowels.
   * @post Prints the consonants from 's', each prefixed with a dot.
   * @invariant 'ch' iterates through each character of the input string.
   */
  for (auto ch : s) {
    ch = tolower(ch); // Functional Utility: Convert the character to lowercase.
    /**
     * @brief Conditional block to check if the character is a vowel.
     * @pre 'ch' is the current lowercase character.
     */
    if (vwl.find(ch) != vwl.end()) // If the character is found in the vowel set.
      continue;                     // Skip this character (it's a vowel).
    cout << '.' << ch;            // Otherwise (it's a consonant), print a dot followed by the character.
  }
  cout << endl; // Print a newline at the end of the output.
  return 0;     // Return 0 for successful execution.
}
