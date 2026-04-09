/**
 * @file 236A.CPP
 * @brief This program classifies a user as "female" or "male" based on the parity
 * of the count of distinct characters in their username. If the number of distinct
 * characters is even, the user is classified as female; otherwise, male.
 *
 * Algorithm: Character counting using a set and parity check.
 * 1. Read the username as a string.
 * 2. Insert each character of the username into a `std::set` to automatically count unique characters.
 * 3. Check the size of the set. If it's even, print "CHAT WITH HER!".
 * 4. If the size is odd, print "IGNORE HIM!".
 * Time Complexity: O(L log C) where L is the length of the username string and C is the number of distinct characters.
 *                  The `log C` factor comes from the insertion into `std::set`.
 * Space Complexity: O(C) where C is the number of distinct characters, to store unique characters in the set.
 */

#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <map> // Not used, but included in original.
#include <math.h> // Not used, but included in original.
#include <set>
#include <stdio.h> // Not used, but included in original.
#include <string>

using namespace std;

/**
 * @brief Main function of the program.
 * Reads the username, determines the gender classification based on distinct character count parity, and prints the result.
 * @pre The input `username` is a string.
 * @post Prints "CHAT WITH HER!" if the username is classified as female, "IGNORE HIM!" otherwise.
 * @return 0 upon successful execution.
 */
int main() {
  string username;
  // Functional Utility: Reads the username from standard input.
  cin >> username;
  // Functional Utility: A set to store unique characters from the username. `std::set` automatically handles uniqueness.
  set<char> letters;
  // Block Logic: Iterates through the username string to populate the set with unique characters.
  // Invariant: After each iteration, `letters` contains all unique characters encountered up to the current index.
  for (auto i = 0; i < username.length(); i++) {
    letters.insert(username[i]);
  }
  // Block Logic: Checks the parity of the count of distinct characters to classify the user.
  // Pre-condition: `letters.size()` returns the count of unique characters in the username.
  if (letters.size() % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
  return 0;
}
