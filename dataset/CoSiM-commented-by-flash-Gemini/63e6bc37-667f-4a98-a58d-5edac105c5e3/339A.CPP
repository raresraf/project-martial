/**
 * @63e6bc37-667f-4a98-a58d-5edac105c5e3/339A.CPP
 * @brief This program solves Codeforces problem 339A, "Helpful Maths".
 * It takes a string representing a sum of numbers (e.g., "3+2+1") where
 * numbers are single digits 1, 2, or 3, separated by '+'. The goal is to
 * rearrange the numbers in non-decreasing order while preserving the '+' signs.
 *
 * Algorithm: String manipulation. First, remove all '+' signs to isolate the digits.
 * Then, sort the digits. Finally, re-insert the '+' signs between the sorted digits.
 * Time Complexity: O(L log L) due to sorting, where L is the number of digits.
 * Space Complexity: O(L) for storing the string.
 */
#include <algorithm> // For std::sort.
#include <cstring>   // Included but not explicitly used in this snippet.
#include <iostream>  // For standard input/output operations (cin, cout).
#include <string>    // For using std::string.
using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to reorder a sum string.
 * Reads a sum string, extracts digits, sorts them, and reconstructs the string.
 * @return 0 on successful execution.
 */
int main() {
  string s; // Declares a string variable to store the input sum.
  cin >> s; // Reads the sum string from standard input.
  
  /**
   * Block Logic: Removes all '+' characters from the input string 's'.
   * This prepares the string to contain only digits, which can then be sorted.
   * Invariant: After this loop, 's' contains only the digits from the original sum.
   */
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '+') { // Inline: If the current character is a '+'.
      s.erase(s.begin() + i); // Erases the '+' character.
    }
  }
  
  // Sorts the string 's' lexicographically, which for digits effectively sorts them numerically.
  sort(s.begin(), s.end()); 
  
  int len = s.length(); // Gets the new length of the string (now containing only digits).
  int it = 1;           // Iterator/index for inserting '+' signs. Initialized to 1 to insert after the first digit.
  
  /**
   * Block Logic: Re-inserts '+' signs between the sorted digits.
   * Iterates through the sorted digits (excluding the last one) and inserts '+'
   * at appropriate positions, adjusting the insertion point (`it`) accordingly.
   * Invariant: After this loop, 's' contains the sorted digits separated by '+' signs.
   */
  for (int i = 0; i < len - 1; i++) {
    s.insert(it, "+"); // Inserts a '+' character at position `it`.
    it += 2;           // Moves `it` two positions forward (past the newly inserted '+' and the next digit).
  }
  cout << s; // Prints the reordered sum string.
  return 0;  // Indicates successful program termination.
}
