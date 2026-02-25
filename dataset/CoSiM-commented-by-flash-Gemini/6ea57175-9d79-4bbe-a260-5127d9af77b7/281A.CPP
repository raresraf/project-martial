/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem from Codeforces (Problem 281A).
 *        This program reads a single word as input, capitalizes its first letter,
 *        and then prints the modified word.
 * Algorithm: String Manipulation / Character Case Conversion
 * Time Complexity: O(L), where L is the length of the input string for reading.
 *                  The capitalization operation itself is O(1).
 * Space Complexity: O(L) to store the input string.
 */

#include <bits/stdc++.h> // Includes most standard C++ libraries, including iostream, string, and cctype.
#define ll long long int      ///< @brief Defines 'll' as a shorthand for long long int.
#define loop(k) for (i = 0; i < k; ++i) ///< @brief Macro for a common for-loop pattern. (Note: 'i' is undeclared in this scope)
#define loop2(k, l) for (j = k; j < l; ++j) ///< @brief Macro for another common for-loop pattern. (Note: 'j' is undeclared in this scope)
#define mod 1000000007 ///< @brief A common modulo value used in problems involving large numbers.
using namespace std;

/**
 * @brief Main function to capitalize the first letter of an input word.
 * @return 0 if the program executes successfully.
 */
int main() {
  // Functional Utility: Disables synchronization between C-style I/O and C++ streams.
  // This helps speed up I/O operations in competitive programming.
  std::ios_base::sync_with_stdio(false); 
  // cin.tie(NULL); // Commented out: This line would untie cin from cout for further I/O optimization.
  
  // Declare variables:
  // n, i, j, t: declared but not used in this specific problem logic.
  // s: string to store the input word.
  ll n = 1, i, j, t; 
  string s;
  
  // Read the input word into string 's'.
  cin >> s;
  
  // Block Logic: Capitalize the first character of the string.
  // Pre-condition: 's' contains the input word.
  // Post-condition: The first character of 's' is converted to its uppercase equivalent.
  s[0] = toupper(s[0]); // Convert the first character of the string to uppercase.
  
  // Print the modified string.
  cout << s;
  // Print a newline character.
  cout << "\n";
  return 0; // Indicate successful program termination.
}