/**
 * @file 266A.CPP
 * @brief Solves the "Stones on the Table" problem.
 * @details The problem asks for the minimum number of stones to remove from a row
 * of colored stones such that no two adjacent stones have the same color. This program
 * achieves this by iterating through the stones and counting every time a stone has the
 * same color as the one immediately preceding it.
 *
 * @algorithm Single-pass counting of adjacent duplicates.
 * @complexity
 * Time complexity: O(N), where N is the number of stones, as it makes a single pass.
 * Space complexity: O(1), as it only stores the color of the previous stone.
 */
#include <iostream>
#include <string>
using namespace std;
int main() {
  // n: total number of stones.
  // count: number of stones to remove.
  int n = 0, count = 0;
  cin >> n;
  
  // so: "stone old", stores the color of the previous stone.
  // sn: "stone new", stores the color of the current stone being read.
  char so, sn;
  
  // Read the first stone's color to initialize the comparison.
  cin >> so;
  n = n - 1; // Decrement n as one stone has been processed.
  
  /**
   * @brief Loop through the remaining stones to count duplicates.
   * @invariant Iterates through the `n-1` remaining stones.
   */
  while (n--) {
    cin >> sn;
    /**
     * @brief Compare the current stone with the previous one.
     * @post If the colors are the same, increment the removal counter.
     */
    if (so == sn)
      count++;
    // The current stone becomes the "old" stone for the next iteration.
    so = sn;
  }
  
  // Print the total number of stones to remove.
  cout << count << endl;
}