/**
 * @file 339A.CPP
 * @brief Reorders numbers in a mathematical sum string in non-decreasing order.
 *
 * Algorithm: Extracts numbers from the string, sorts them, and then reconstructs the sum string.
 * Time Complexity: O(L log N), where L is the length of the input string and N is the count of numbers.
 *                  The sorting dominates if L is not significantly larger than N log N.
 * Space Complexity: O(N) to store the extracted numbers.
 */
#include <algorithm> // Required for std::sort.
#include <iostream>  // Required for input/output operations (cin, cout).
#include <string>    // Required for std::string.
#include <vector>    // Required for std::vector.

using namespace std;

/**
 * @brief Custom comparison function for sorting characters in non-decreasing order.
 * @param i The first character to compare.
 * @param j The second character to compare.
 * @return True if 'i' should come before 'j' (i.e., 'i' is less than 'j').
 * Functional Utility: Ensures std::sort arranges characters representing numbers in ascending order.
 */
bool desc(char i, char j) { return i < j; } // Note: 'desc' is a misnomer, it sorts in ascending order.

/**
 * @brief Main function to reorder numbers in a sum string.
 * @pre Input is a string containing numbers separated by '+' signs (e.g., "1+3+2").
 * @post Prints the sum string with numbers reordered in non-decreasing order (e.g., "1+2+3").
 */
int main() {
  string kata; // Variable to store the input sum string.
  // Functional Utility: Reads the input sum string from standard input.
  cin >> kata;

  vector<char> coins; // Vector to store the extracted numbers as characters.

  /**
   * Block Logic: Iterates through the input string to extract numerical characters.
   * Invariant: In each iteration, if the character is not a '+', it's a number and is added to the 'coins' vector.
   */
  for (int i = 0; i < kata.length(); i++) {
    if (kata[i] != '+') {
      coins.push_back(kata[i]);
    }
  }

  // Functional Utility: Sorts the extracted numerical characters in non-decreasing order.
  sort(coins.begin(), coins.end(), desc);

  /**
   * Block Logic: Prints the sorted numbers, reinserting '+' signs between them.
   * Invariant: The first number is printed without a preceding '+'. Subsequent numbers are preceded by a '+'.
   */
  for (int i = 0; i < coins.size(); i++) {
    if (i != 0) { // Pre-condition: Checks if it's not the first number.
      cout << "+";
    }
    cout << coins[i]; // Prints the current number.
  }
  // No explicit return 0; is needed for main in C++ if it reaches the end.
  // However, for competitive programming, it's often good practice to include it.
}
