/**
 * @file 112A.CPP
 * @brief Solution for the "Petya and Strings" programming problem.
 * @details This program performs a case-insensitive lexicographical comparison of two input strings.
 * It iterates through both strings simultaneously, converting each character to uppercase to ignore
 * case differences. The first pair of characters that do not match determines the lexicographical
 * order. The program outputs -1 if the first string is smaller, 1 if it is larger, and 0 if they
 * are equal.
 *
 * @algorithm Case-insensitive lexicographical string comparison.
 * @complexity
 * Time Complexity: O(L), where L is the length of the strings. The loop runs at most L times,
 * stopping at the first differing character.
 * Space Complexity: O(L), for storing the two input strings.
 */
#include <iostream>
#include <string>
#include <cctype> // Recommended for explicit inclusion of toupper

int main() {
  // Functional Utility: Decouples C++ streams from C streams and unties cin from cout
  // to optimize I/O performance, which is a common practice in competitive programming.
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(NULL);
  std::string str1, str2;
  std::cin >> str1 >> str2;

  bool isSame = true;
  // Block Logic: Iterate through the strings to find the first point of difference.
  for (size_t i = 0; i < str1.length(); i++) {
    // Convert characters to a common case (uppercase) for case-insensitive comparison.
    char a = toupper(str1[i]), b = toupper(str2[i]);
    
    // Pre-condition: Check if the characters at the current position are different.
    if (a != b) {
      isSame = false;
      // The first differing character determines the lexicographical order of the entire string.
      std::cout << ((a > b) ? "1" : "-1");
      break; // Exit the loop as the result is found.
    }
  }

  // This block is executed only if the loop completes without finding any differences.
  if (isSame) {
    std::cout << "0";
  }
  
  return 0;
}
