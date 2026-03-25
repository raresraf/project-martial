#include <iostream>
#include <string>   // For std::string
#include <cctype>   // For toupper()

/**
 * @file 281A.CPP
 * @brief Solution to a string manipulation problem, "Word Capitalization".
 *
 * Problem Statement:
 * Given a word, capitalize its first letter. That is, change its first
 * letter to uppercase if it is lowercase, and leave it unchanged otherwise.
 *
 * Algorithm:
 * The program reads a word into a C-style character array. It then checks if
 * the first character is a lowercase letter by comparing its ASCII value.
 * If it is, it converts it to uppercase by subtracting 32 from its ASCII value.
 *
 * A more modern C++ approach would be to use `std::string` and the
 * `toupper()` function from the `<cctype>` header.
 *
 * Time Complexity: O(L) for I/O, where L is the length of the word. The
 * capitalization logic itself is O(1).
 * Space Complexity: O(L) to store the word.
 */

using namespace std;

int main() {
  // Using a C-style character array to store the word.
  char c[1001]; // Increased size slightly for safety.
  cin >> c;

  // Pre-condition: Check if the first character is a lowercase letter.
  // The ASCII value of 'a' is 97. This checks for 'a' through 'z'.
  if (c[0] >= 'a' && c[0] <= 'z') {
    // Convert the first character to uppercase.
    // The difference between a lowercase and uppercase letter in ASCII is 32.
    // A more idiomatic way is: c[0] = toupper(c[0]);
    c[0] -= 32;
  }

  // Print the resulting word.
  cout << c << endl;

  return 0;
}