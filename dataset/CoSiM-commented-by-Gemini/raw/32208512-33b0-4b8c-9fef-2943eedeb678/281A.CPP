/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem.
 * @details This program takes a single word as input and capitalizes its first letter,
 * leaving the rest of the word unchanged.
 *
 * @algorithm The algorithm is a direct string manipulation:
 * 1. Read the input word into a `std::string`.
 * 2. Access the character at index 0.
 * 3. Convert this character to its uppercase equivalent using `toupper()`.
 * 4. The string is modified in-place.
 * 5. Print the resulting capitalized word.
 *
 * @time_complexity O(L), where L is the length of the word, dominated by the string
 * reading and printing operations. The capitalization itself is O(1).
 * @space_complexity O(L), to store the input word.
 */
#include <ctype.h>
#include <iostream>
using namespace std;

int main() {
  string a;
  // Read the word from standard input.
  cin >> a;
  // Access the first character and convert it to uppercase in-place.
  a[0] = toupper(a[0]);
  // Print the modified string.
  cout << a;
  return 0;
}
