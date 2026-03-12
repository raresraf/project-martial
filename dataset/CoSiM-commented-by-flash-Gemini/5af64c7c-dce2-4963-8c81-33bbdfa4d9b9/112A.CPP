/**
 * @5af64c7c-dce2-4963-8c81-33bbdfa4d9b9/112A.CPP
 * @brief This program solves problem 112A ("Petya and Strings").
 * It performs a case-insensitive lexicographical comparison of two input strings.
 * The program outputs -1 if the first string is lexicographically smaller,
 * 1 if it is larger, and 0 if both strings are equal.
 *
 * Algorithm: Two-pass approach: first pass converts strings to lowercase, second pass compares characters.
 * Time Complexity: O(L) where L is the length of the input strings.
 * Space Complexity: O(L) for storing the strings.
 */

#include <bits/stdc++.h> // Includes many standard C++ library headers.
#define ll long long int // Defines 'll' as a shorthand for long long int.
#define loop(k) for (i = 0; i < k; ++i) // Macro for a common for-loop pattern.
#define loop2(k, l) for (j = k; j < l; ++j) // Macro for another for-loop pattern.
#define mod 1000000007 // Defines 'mod' for a common modulo value.
using namespace std;   // Uses the standard namespace.

int main() {
  // Block Logic: Optimizes C++ standard I/O operations.
  // `std::ios_base::sync_with_stdio(false)` unties C++ streams from C stdio.
  // `cin.tie(NULL)` unties cin from cout, preventing flushing of cout before cin operations.
  std::ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  ll t = 1, i = 0, j = 0, k = 0; // Declares variables; 't' is for test cases (runs once), 'i' is loop counter.
  // cin>>t; // Commented out, so it runs for a single test case.
  /**
   * Block Logic: Processes test cases. For this snippet, it runs once.
   */
  while (t--) { 
    string s, s1; // Declares two string variables to store the input words.
    cin >> s >> s1; // Reads the two strings from standard input.
    int len = s.length(); // Gets the length of the first string (assumed same for both).
    
    /**
     * Block Logic: Converts both input strings to lowercase for case-insensitive comparison.
     * Invariant: Both 's' and 's1' contain their lowercase equivalents.
     */
    loop(len) { // Loop from i = 0 to len-1 using the 'loop' macro.
      s[i] = tolower(s[i]);   // Inline: Converts character s[i] to lowercase.
      s1[i] = tolower(s1[i]); // Inline: Converts character s1[i] to lowercase.
    }
    
    /**
     * Block Logic: Performs character-by-character lexicographical comparison of the lowercase strings.
     * Invariant: The program outputs -1, 1, or 0 and terminates as soon as a difference is found or strings are equal.
     */
    loop(len) { // Loop from i = 0 to len-1 using the 'loop' macro.
      if (s[i] > s1[i]) { // Inline: If character in 's' is lexicographically greater.
        cout << "1";      // Output 1.
        break;            // Terminate loop.
      } else if (s[i] < s1[i]) { // Inline: If character in 's' is lexicographically smaller.
        cout << "-1";         // Output -1.
        break;                // Terminate loop.
      }
    }
    // Block Logic: If the comparison loop completed without a 'break', it means both strings are identical.
    if (i == len) // Inline: If loop completed, 'i' will be equal to 'len'.
      cout << "0"; // Output 0.
    cout << "\n"; // Prints a newline character.
  }
  return 0; // Indicates successful program termination.
}