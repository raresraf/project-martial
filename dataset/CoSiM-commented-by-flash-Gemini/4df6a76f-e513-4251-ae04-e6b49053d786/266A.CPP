/**
 * @file 266A.CPP
 * @brief Counts the minimum number of stones to remove from a row such that no two adjacent stones are of the same color.
 *
 * Problem Name: Stones on the Table (Codeforces 266A)
 * Algorithm: Iterative direct comparison of adjacent characters. For each pair of adjacent stones, if they are identical,
 *            it increments a counter, representing a stone that needs to be removed.
 * Time Complexity: O(L) where L is the length of the input string, due to a single linear pass.
 * Space Complexity: O(L) for storing the input string.
 */
/*
 *Link: https://codeforces.com/problemset/problem/266/A
 */

#include <iostream> // Required for input/output operations (cin, cout).
#include <string>   // Required for std::string manipulations.
using namespace std;

/**
 * @brief Main function to count adjacent identical stones.
 * @pre Input consists of an integer 'n' (number of stones), followed by a string 'stones' representing their colors.
 * @post Prints the total count of adjacent identical stone pairs.
 */
int main() {
  string stones; // Variable to store the input string of stone colors.
  int n, answer = 0; // 'n': number of stones, 'answer': counter for adjacent identical stones.

  // Functional Utility: Reads the number of stones 'n' and the string of 'stones'.
  // Note: 'n' is read but `stones.length()` is used, which is robust.
  cin >> n >> stones;

  /**
   * Block Logic: Iterates through the string, comparing each stone with its immediate successor.
   * Invariant: `answer` increments whenever `stones[i]` is identical to `stones[i+1]`, indicating a stone to be removed.
   */
  for (int i = 0; i < stones.length() - 1; i++) {
    /**
     * Block Logic: Checks if the current stone is the same color as the next stone.
     * Pre-condition: `stones[i]` and `stones[i+1]` are valid characters in the string.
     * Invariant: If they are identical, `answer` is incremented.
     */
    if (stones[i] == stones[i + 1])
      answer++; // Increments count if adjacent stones are identical.
  }

  // Functional Utility: Prints the final count of stones to be removed.
  cout << answer << endl;
  // No explicit return 0; is needed for main in C++ if it reaches the end.
  // However, for competitive programming, it's often good practice to include it.
}