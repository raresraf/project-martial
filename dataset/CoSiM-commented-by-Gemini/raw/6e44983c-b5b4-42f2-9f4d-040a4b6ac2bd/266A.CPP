
// link : https://codeforces.com/problemset/problem/266/A

#include <bits/stdc++.h>

using namespace std;

/**
 * @file 266A.CPP
 * @brief Solution to Codeforces problem 266A, "Stones on the Table".
 * @details The program calculates the minimum number of stones to remove from a
 * row so that no two adjacent stones have the same color.
 *
 * Algorithm:
 * This implementation simulates the physical removal of stones. It reads the
 * stone colors into a character vector. It then iterates through the vector,
 * and if it finds two adjacent identical stones, it erases the first one from
 * the vector. To handle the change in the vector's size and contents, the loop
 * index and size variable are decremented. This process is repeated until the
 * entire vector has been checked. A counter tracks the number of erasures,
 * which corresponds to the number of stones removed.
 *
 * Time Complexity: O(N^2) in the worst case (e.g., a string of all identical
 * characters), because `vector::erase` is an O(N) operation which may be
 * performed up to N times.
 * Space Complexity: O(N) to store the vector of characters.
 */

string s;

// Global variables `a`, `b`, `c` are used for input size, vector size, and removal count.
int a, b, c;

vector<char> v;

main() {
  // Read the number of stones.
  cin >> a;

  // Read each stone color and populate the vector.
  for (int i = 0; i < a; i++) {
    char r;
    cin >> r;
    v.push_back(r);
  }

  b = v.size();

  /**
   * @brief Main simulation loop.
   * Iterates through the vector, removing stones that match their right neighbor.
   */
  for (int i = 0; i < b - 1; i++) {
    // Check if the current stone has the same color as the next one.
    if (v[i] == v[i + 1]) {
      // If they match, remove the current stone.
      v.erase(v.begin() + i);
      
      // Decrement loop index to re-check the current position, as a new
      // element has shifted into index `i`.
      i--;
      // Decrement the size variable to reflect the change in the vector.
      b--;
      // Increment the count of removed stones.
      c++;
    }
  }

  cout << c;
}
