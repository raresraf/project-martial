/**
 * @file 266A.CPP
 * @brief Solves the "Stones on the Table" problem from Codeforces (266A).
 * @details The program calculates the minimum number of stones that need to be
 * removed from a row so that no two adjacent stones have the same color.
 *
 * @section Algorithm
 * The implemented solution works as follows:
 * 1. Reads the number of stones, 'n', and a string representing their colors.
 * 2. Copies the stones into a `std::vector<char>`.
 * 3. Iterates through the vector, comparing each stone with the one next to it.
 * 4. If two adjacent stones have the same color, a `removals` counter is
 *    incremented, and the second stone is erased from the vector using `vector::erase`.
 * 5. This process continues until the end of the (potentially modified) vector is reached.
 * 6. Finally, the total number of removals is printed.
 *
 * @section Complexity
 * - Time Complexity: O(n^2). The use of `vector::erase` inside a loop is inefficient.
 *   Erasing an element from a vector takes time proportional to the number of
 *   elements that must be shifted, leading to a quadratic overall complexity.
 * - Space Complexity: O(n) to store the stones in the vector.
 *
 * @note An optimal solution with O(n) time complexity can be achieved by simply
 * iterating through the original string and counting `s[i] == s[i+1]` without
 * modifying the data structure.
 */
// http://codeforces.com/problemset/problem/266/A
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  string input;
  cin >> n;
  cin >> input;

  // The input string is copied into a vector of characters.
  vector<char> stones;
  for (int i = 0; i < n; i++)
    stones.push_back(input[i]);
  
  // 'removals' will count the number of stones taken.
  int removals = 0;

  /**
   * @brief Block Logic: Iterates through the vector to find and remove adjacent duplicates.
   * @invariant The iterator 'itr' moves through the vector. If an element is erased,
   * the loop structure correctly handles the modified vector bounds but at a high
   * performance cost.
   */
  for (auto itr = stones.begin(); itr < stones.end() - 1; itr++) {
    /**
     * @brief Logic: If a stone has the same color as its neighbor, the neighbor
     * is removed and the count is incremented. This `erase` operation is O(n)
     * and makes the overall algorithm inefficient.
     */
    if (*itr == *(itr + 1)) {
      stones.erase(itr + 1);
      removals++;
    }
  }
  
  // Output the total number of removals required.
  cout << removals << endl;
  return 0;
}