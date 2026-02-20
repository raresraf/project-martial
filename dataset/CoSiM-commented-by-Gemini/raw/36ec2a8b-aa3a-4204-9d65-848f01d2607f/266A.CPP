/**
 * @file 266A.CPP
 * @brief Solves the "Stones on the Table" problem using the C++ STL `unique` algorithm.
 *
 * This program calculates the minimum number of stones to remove from a row
 * so that no two adjacent stones have the same color. This version uses a
 * highly idiomatic C++ approach.
 *
 * Algorithm:
 * 1. Read the number of stones 'n' and the string 's' representing their colors.
 * 2. Use `std::unique` to reorder the string in-place, moving all unique
 *    consecutive elements to the beginning. `unique` returns an iterator to the
 *    end of this new unique range. For "RRGGB", this would result in "RGB" at
 *    the start of the string.
 * 3. Use the erase-remove idiom (`s.erase(unique(...), s.end())`) to truncate
 *    the string, removing the leftover duplicate elements.
 * 4. The number of stones to remove is the difference between the original
 *    length 'n' and the new length of the modified string.
 *
 * Time Complexity: O(N), where N is the length of the string, as `std::unique`
 * and `s.erase` both have linear complexity.
 * Space Complexity: O(N) to store the string.
 */
#include <algorithm> // For std::unique
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
using namespace std;
int main() {
  int n; // Original number of stones.
  cin >> n;
  string s; // String representing the colors of the stones.
  cin >> s;

  /**
   * Block Logic: Erase-Remove Idiom with std::unique.
   * `std::unique` shifts adjacent unique elements to the front of the string
   * and returns an iterator to the new logical end.
   * `s.erase()` then removes the now-unused elements from this new end
   * to the original end of the string.
   */
  s.erase(unique(s.begin(), s.end()), s.end());
  
  // The number of removed stones is the original length minus the length
  // of the string after collapsing adjacent duplicates.
  cout << n - s.length() << endl;
}
