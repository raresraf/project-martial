/**
 * @5e913431-be1e-4a0c-add6-45e50860787e/266A.CPP
 * @brief This program solves problem 266A, "Stones on the Table".
 * Given a string `row` representing a sequence of colored stones, the objective
 * is to find the minimum number of stones to remove such that no two adjacent
 * stones have the same color. This is achieved by iteratively removing duplicate
 * adjacent stones.
 *
 * Algorithm: Two-pointer (iterator-based) linear scan with in-place character removal.
 * Time Complexity: O(N * L) where N is the length of the string and L is the average
 *                  length of the string during `erase` operations. In the worst case
 *                  (e.g., all characters are the same), it can degrade to O(N^2)
 *                  due to repeated `std::string::erase` operations.
 * Space Complexity: O(N) for storing the string.
 */

#include <iostream> // For standard input/output operations (cin, cout).
#include <string>   // For using std::string.
#include <iterator> // For std::string::iterator (implicitly included with string).

using namespace std; // Uses the standard namespace.

int main() {
  int n; // Declares 'n' for the length of the string (read but not directly used in the loop bounds).
  int numberOfRemoves = 0; // Initializes a counter for removed stones.
  string row; // Declares a string variable to store the input sequence.

  cin >> n; // Reads the integer 'n' (length of the string).
  cin.get(); // Consumes the newline character left in the input buffer after `cin >> n`.
  getline(cin, row); // Reads the entire line into 'row', including spaces if any.

  // Inline: Initializes 'previous' iterator to the first character and 'next' iterator to the second.
  string::iterator previous = row.begin(); 
  string::iterator next = row.begin() + 1;

  /**
   * Block Logic: Iterates through the string using two iterators to find and remove
   * adjacent identical characters.
   * Invariant: `numberOfRemoves` correctly counts the stones removed to eliminate adjacent duplicates.
   */
  while (next != row.end()) { // Loop continues as long as 'next' is within the string bounds.
    // Inline: Checks if the characters pointed to by 'previous' and 'next' are identical.
    if (*previous == *next) { 
      // Block Logic: If adjacent characters are the same, remove the duplicate.
      // `row.erase(next)` removes the character and returns an iterator to the
      // element that followed the erased one. This new iterator is assigned back to `next`.
      // `previous` remains pointing to the same logical character (now the character after the erased one).
      next = row.erase(next); 
      numberOfRemoves++; // Increments the counter for removed stones.
    } else {
      // Block Logic: If adjacent characters are different, advance both iterators.
      previous++; // Moves 'previous' to the next character.
      next++;     // Moves 'next' to the character after 'previous'.
    }
  }

  cout << numberOfRemoves; // Prints the total number of stones removed.

  return 0; // Indicates successful program termination.
}