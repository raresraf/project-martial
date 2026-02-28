/**
 * @file 236A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 236A - "Boy or Girl").
 *
 * This program determines a gender-like classification ("Boy" or "Girl") for a username
 * based on the parity of the number of unique characters it contains.
 * If the number of unique characters is odd, the output is "IGNORE HIM!";
 * if even, the output is "CHAT WITH HER!".
 *
 * Algorithm: Character counting and parity check.
 * It uses a `std::set` to efficiently count the number of distinct characters in the input string.
 * The size of the set then determines the parity, which dictates the output message.
 *
 * Time Complexity: O(L), where L is the length of the input string.
 * This is because inserting each character into a `std::set` takes logarithmic time
 * with respect to the set's size (which is at most 26 for English alphabet characters),
 * making the overall complexity approximately linear.
 * Space Complexity: O(K), where K is the number of unique characters in the string.
 * In the context of English alphabet, K is at most 26, making it effectively O(1).
 */
// author @Nishant

#include <bits/stdc++.h> // Includes various standard libraries for C++.
using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads a username, counts its unique characters, and prints a message based on the count's parity.
 */
int main() {
  string s; // Variable to store the input username.
  // Pre-condition: Read the input username string.
  cin >> s;
  // Functional Utility: A set automatically stores only unique elements.
  set<char> un; // Set to store unique characters from the input string.
  // Block Logic: Iterate through each character of the input string and insert it into the set.
  // Invariant: 'x' represents the current character being processed.
  for (char x : s) {
    un.insert(x); // Inserting into a set ensures only unique characters are stored.
  }
  // Block Logic: Check the parity of the number of unique characters.
  if (un.size() % 2) { // If the number of unique characters is odd.
    cout << "IGNORE HIM!" << endl; // Print message for "Boy".
  } else {             // If the number of unique characters is even.
    cout << "CHAT WITH HER!" << endl; // Print message for "Girl".
  }
  return 0; // Indicate successful execution.
}