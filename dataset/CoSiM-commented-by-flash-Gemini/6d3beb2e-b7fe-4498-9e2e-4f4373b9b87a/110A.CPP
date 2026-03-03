/**
 * @file 110A.CPP
 * @brief Solution for Codeforces Problem 110A - "Lucky Numbers".
 * This program determines if a given number is "nearly lucky".
 * A number is nearly lucky if the count of "lucky digits" (4 and 7) in its
 * decimal representation is itself a single-digit lucky number (i.e., 4 or 7).
 * Algorithm: Count lucky digits in the input string and then check if this count is 4 or 7.
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(L) - to store the input string.
 */
#include <iostream> // Required for standard input/output operations (cin, cout).
#include <string>   // Required for std::string.

using namespace std; // Uses the standard namespace.

int main() {
  string s; // Declares a string to store the input number.
  cin >> s; // Reads the input number as a string.
  
  int sz = s.length(); // Gets the length of the input string.
  int cnt = 0;         // Counter for lucky digits (4 or 7).
  
  /**
   * Block Logic: Iterates through each character of the input string to count lucky digits.
   * Invariant: `cnt` holds the number of lucky digits encountered so far.
   */
  for (int i = 0; i < sz; ++i) { // Loop through each character.
    // Checks if the current character is '4' or '7'.
    if (s[i] == '4' || s[i] == '7')
      cnt++; // Increment the lucky digit counter.
  }

  // Decision Logic: Determines if the number is "nearly lucky" based on the count of lucky digits.
  // A number is "nearly lucky" if `cnt` is either 4 or 7.
  if (cnt == 4 || cnt == 7)
    cout << "YES\n"; // Output "YES" if `cnt` is 4 or 7.
  else
    cout << "NO\n";  // Output "NO" otherwise.

  return 0; // Indicates successful program execution.
}
