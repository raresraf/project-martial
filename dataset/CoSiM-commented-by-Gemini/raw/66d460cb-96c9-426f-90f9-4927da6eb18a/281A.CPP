#include <iostream>

using namespace std;

/**
 * @file 281A.CPP
 * @brief Concise solution to the "Word Capitalization" problem from Codeforces (Problem 281A).
 * @details This program reads a word and capitalizes its first letter directly in-place.
 *
 * Algorithm:
 * The program reads a word from standard input into a string. It then accesses the
 * first character of the string by its index (0) and applies the `toupper()` function
 * to convert it to uppercase. The change is made directly to the string in memory.
 * Finally, the entire modified string is printed to the standard output.
 *
 * Time Complexity: O(L), where L is the length of the string. This is primarily
 * due to the time taken to read and print the string.
 * Space Complexity: O(L), to hold the string in memory.
 */
int main() {
  string c;
  // Read the input word.
  cin >> c;

  // Capitalize the first character of the string in-place.
  c[0] = toupper(c[0]);

  // Print the modified string.
  cout << c;

  return 0;
}
