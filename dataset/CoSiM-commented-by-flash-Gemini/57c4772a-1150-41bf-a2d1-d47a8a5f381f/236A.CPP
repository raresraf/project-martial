/**
 * @57c4772a-1150-41bf-a2d1-d47a8a5f381f/236A.CPP
 * @brief This program determines an output phrase based on the parity (even or odd)
 * of the number of unique characters in an input string. It uses a `std::map`
 * to efficiently count unique characters.
 *
 * Algorithm: Character frequency counting using a hash map, followed by a parity check.
 * Time Complexity: O(L log C) where L is the length of the input string and C is the number of unique characters.
 *                  Practically O(L) for small, fixed character sets due to map operations.
 * Space Complexity: O(C) for the map, effectively O(1) for small, fixed character sets.
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes many standard libraries.

using namespace std; // Uses the standard namespace to avoid prefixing std::.

int main() {

  string s; // Declares a string variable to store the input username.
  cin >> s; // Reads the username string from standard input.

  // Declares a map to store the frequency of each character.
  // The keys are characters, and values are their counts.
  map<char, int> m; 

  /**
   * Block Logic: Iterates through each character of the input string to populate the map
   * with character frequencies. This implicitly counts unique characters as well.
   * Invariant: After each iteration, 'm' accurately reflects the frequencies of characters
   * encountered so far.
   */
  for (int i = 0; i < (int)s.length(); i++) {
    // Inline: Increments the count for the character s[i]. If the character is not
    // yet in the map, it's added with a count of 1.
    m[s[i]]++; 
  }

  /**
   * Block Logic: Determines the output phrase based on the parity of the number of unique characters.
   * Pre-condition: 'm' contains the frequencies of all characters in the input string.
   * Invariant: One of two specific phrases is printed based on whether the number of unique
   * characters ('m.size()') is even or odd.
   */
  // Inline: `m.size()` gives the number of unique characters. `& 1` performs a bitwise AND
  // operation to check if the number is odd (result 1) or even (result 0).
  // The ternary operator then selects the appropriate phrase.
  cout << (m.size() & 1 ? "IGNORE HIM!" : "CHAT WITH HER!");
  
  // The program implicitly returns 0 upon successful completion of main.
}