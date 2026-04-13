/**
 * @file 236A.CPP
 * @brief This program determines a message ("CHAT WITH HER!" or "IGNORE HIM!")
 * based on the parity (odd or even) of the number of unique characters in an input string (username).
 * Algorithm: Counts unique characters by iterating through the string and marking duplicates.
 * Time Complexity: O(N^2) due to nested loops for checking character uniqueness, where N is the string length.
 * Space Complexity: O(N) for storing the input string.
 */
#include <cstring> // Required for strlen.
#include <ctype.h> // Not directly used in this code, but included in original.
#include <iomanip> // Not directly used in this code, but included in original.
#include <iostream> // Required for cin, cout.
#include <math.h> // Not directly used in this code, but included in original.
#include <time.h> // Not directly used in this code, but included in original.
using namespace std;

int main() {
  char array[105]; // Buffer to store the input username string, max length 104 + null terminator.
  int m; // Flag to ensure 'n' is decremented only once per unique character.
  // Reads the input username string.
  cin >> array;
  int n = strlen(array); // Initializes 'n' with the total length of the string, which will be decremented for duplicate characters.
  /**
   * Block Logic: Iterates through the string to find and count unique characters.
   * For each character, it checks subsequent characters for duplicates. If a duplicate is found,
   * 'n' (representing the count of unique characters) is decremented.
   * Invariant: 'n' holds the count of characters that are either unique or the first occurrence of a duplicate.
   */
  for (int i = 0; i < strlen(array); i++) {
    m = 0; // Resets the duplicate flag for the current character.
    /**
     * Block Logic: Checks for duplicates of 'array[i]' among the subsequent characters.
     * If a duplicate is found and 'm' is 0, it means this is the first time a duplicate
     * of 'array[i]' has been identified, so 'n' is decremented.
     */
    for (int j = i + 1; j < strlen(array); j++) {
      if (array[i] == array[j] && (m == 0)) {
        n--; // Decrement 'n' as a duplicate of 'array[i]' has been found.
        m++; // Set the flag to prevent further decrements for this 'array[i]'.
      }
    }
  }

  /**
   * Conditional Logic: Determines the output message based on the parity of 'n'.
   * If 'n' (the count of effectively unique characters) is odd, it prints "IGNORE HIM!".
   * If 'n' is even, it prints "CHAT WITH HER!".
   */
  if (n % 2)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";
  // Standard program exit.
  return 0;
}
