/**
 * @file 266A.CPP
 * @brief Solution to Codeforces Problem 266A: "Stones on the Table".
 * Counts the minimum number of stones to remove to make all adjacent stones different colors.
 * Algorithm: Iterative single-pass scan with in-place removal of duplicate adjacent stones.
 * Time Complexity: O(N^2) in the worst case due to `vector::erase()` operations, where N is the number of stones.
 * Space Complexity: O(N) to store the stone colors.
 */
#include <bits/h>

using namespace std;

string s; // Not used in this solution, but declared globally.

/**
 * @brief 'a' represents the total number of stones initially on the table.
 * @brief 'b' dynamically tracks the current number of stones after removals.
 * @brief 'c' counts the total number of stones removed.
 */
int a, b, c; 

vector<char> v; // Stores the colors of the stones.

/**
 * @brief Main function to solve the "Stones on the Table" problem.
 * Reads the number of stones and their colors, then calculates and prints
 * the minimum stones to remove to satisfy the condition.
 */
main() {
  cin >> a; // Reads the initial number of stones.

  /**
   * Block Logic: Reads individual stone colors and populates the vector 'v'.
   * Pre-condition: 'a' contains the number of stones.
   * Invariant: After each iteration, 'v' contains 'i+1' stone colors.
   */
  for (int i = 0; i < a; i++) {
    char r; // Temporary variable to store the current stone color.
    cin >> r; 
    v.push_back(r); // Adds the stone color to the vector.
  }

  b = v.size(); // Initializes 'b' with the initial number of stones.
  c = 0; // Initializes the counter for removed stones to zero.

  /**
   * Block Logic: Iterates through the vector of stones to identify and remove adjacent duplicates.
   * Invariant: At the start of each iteration, 'v' represents the current state of stones,
   *            and 'c' holds the count of stones removed so far.
   */
  for (int i = 0; i < b - 1; i++) {
    /**
     * Block Logic: Checks if the current stone and the next stone have the same color.
     * If they do, one of them is removed to eliminate the adjacency.
     */
    if (v[i] == v[i + 1]) {
      v.erase(v.begin() + i); // Removes the current stone, shifting subsequent elements.
      i--; // Decrements 'i' to re-check the stone that moves into the current position.
      b--; // Decrements 'b' as a stone has been removed.
      c++; // Increments the count of removed stones.
    }
  }

  cout << c; // Prints the total number of removed stones.
}
