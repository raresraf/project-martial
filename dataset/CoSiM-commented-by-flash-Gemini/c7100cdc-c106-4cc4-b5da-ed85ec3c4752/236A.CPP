/**
 * @file 236A.CPP
 * @brief Solution for Codeforces Problem 236A - "Boy or Girl".
 * This program determines if a given username (string of lowercase Latin letters)
 * belongs to a "boy" or a "girl" based on the number of distinct characters it contains.
 * If the count of distinct characters is even, it's a "boy" ("CHAT WITH HER!").
 * If the count is odd, it's a "girl" ("IGNORE HIM!").
 * Algorithm: Count distinct characters using a frequency array/set.
 * Time Complexity: O(L), where L is the length of the input username string.
 * Space Complexity: O(1), as the size of the frequency array is fixed (26 for Latin alphabet).
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries
#include <ctype.h>       // Provides character classification functions (though not explicitly used for this problem)
using namespace std;     // Uses the standard namespace to avoid prefixing std::

int main() {
  string s;    // Declares a string to store the input username.
  cin >> s;    // Reads the username string from standard input.
  
  int len = s.size(); // Gets the length of the input string.
  
  // Declares an array 'a' of size 26 (for 'a' through 'z') initialized to zeros.
  // This array acts as a frequency map or a set to track distinct characters.
  // 'count' stores the number of distinct characters found.
  int a[26] = {0}, count = 0;
  
  /**
   * Block Logic: Iterates through the username string to count distinct characters.
   * For each character, it maps it to an index (0-25) and checks if it has been seen before.
   * Invariant: `count` holds the number of unique characters encountered up to the current position `i`.
   */
  for (int i = 0; i < len; i++) {
    int x = (s[i] - 'a'); // Converts the character to its 0-indexed position in the alphabet.
    // If this character has not been seen before (its entry in 'a' is 0).
    if (a[x] == 0) {
      a[x]++;      // Mark this character as seen.
      count++;     // Increment the count of distinct characters.
    }
  }
  
  // Decision Logic: Determines if the user is a "boy" or "girl" based on the parity of distinct characters.
  if (count % 2 == 0) // If the count of distinct characters is even.
    cout << "CHAT WITH HER!"; // Output for a "boy".
  else // If the count of distinct characters is odd.
    cout << "IGNORE HIM!";    // Output for a "girl".
    
  return 0; // Indicates successful program execution.
}
