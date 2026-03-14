/**
 * @632b202e-e82f-4afa-bb48-beedaff9f956/236A.CPP
 * @brief This program solves problem 236A, "Boy or Girl".
 * It reads a username (string) and determines if the number of distinct characters
 * in the username is odd or even. Based on this, it prints a specific message:
 * "IGNORE HIM!" if the count is odd, and "CHAT WITH HER!" if the count is even.
 *
 * Algorithm: Character frequency counting using a `std::set` to store unique characters.
 * Time Complexity: O(L log L) where L is the length of the string, due to `std::set` insertion.
 * Space Complexity: O(D) where D is the number of distinct characters in the string (D <= 26 for English alphabet).
 */
#include <algorithm> // Included but not explicitly used in this snippet.
#include <iostream>  // For standard input/output operations (cin, cout).
#include <set>       // For using std::set to store unique characters.

using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to execute the "Boy or Girl" problem logic.
 * Reads a username, counts its distinct characters, and prints a message.
 * @return 0 on successful execution.
 */
int main() {

  string s; // Declares a string variable to store the input username.
  cin >> s; // Reads the username from standard input.
  set<char> nchars; // Declares a set to store unique characters encountered.
  int i; // Loop counter (unused after loop completion).

  /**
   * Block Logic: Iterates through the input string and inserts each character into a set.
   * A set automatically handles uniqueness, so `nchars.size()` will be the count of distinct characters.
   * Invariant: After this loop, `nchars` contains all unique characters from the input string `s`.
   */
  for (i = 0; i < s.size(); i++) {
    nchars.insert(s[i]); // Inserts the current character into the set.
  }
  
  /**
   * Block Logic: Checks if the number of distinct characters is odd or even and prints the corresponding message.
   * Invariant: One of "IGNORE HIM!" or "CHAT WITH HER!" is printed.
   */
  if (nchars.size() % 2 != 0) // If the number of distinct characters is odd.
    cout << "IGNORE HIM!" << endl; // Prints "IGNORE HIM!".
  else // If the number of distinct characters is even.
    cout << "CHAT WITH HER!" << endl; // Prints "CHAT WITH HER!".

  return 0; // Indicates successful program termination.
}
