/**
 * @file 266A.CPP
 * @brief Solution for Codeforces Problem 266A - "Stones on the Table".
 * This program calculates the minimum number of stones to remove from a row
 * so that no two adjacent stones are of the same color. This is achieved
 * by counting the number of adjacent pairs with identical colors.
 * Algorithm: Single pass iteration.
 * Time Complexity: O(N) - where N is the number of stones (length of the array/string).
 * Space Complexity: O(N) - to store the input string of stones in a character array.
 */
#include <iostream> // Required for standard input/output operations (cin, cout).

using namespace std; // Uses the standard namespace.

int main() {
  int n, c = 0; // n: number of stones, c: counter for stones to remove, initialized to 0.

  cin >> n; // Reads the total number of stones.

  // Declares a C-style character array `s` of size `n` to store the stone colors.
  // This is a Variable Length Array (VLA), which is a C99 feature and an extension
  // in C++. Modern C++ often prefers `std::vector<char>` or `std::string` for dynamic sizes.
  char s[n]; 

  cin >> s; // Reads the string of stone colors into the character array.

  /**
   * Block Logic: Iterates through the stone array to identify adjacent pairs of the same color.
   * If `s[i]` and `s[i+1]` are the same, it means `s[i+1]` must be removed, so `c` is incremented.
   * Invariant: `c` accumulates the count of stones to be removed (i.e., identical adjacent pairs).
   */
  for (int i = 0; i < n - 1; i++) { // Loop up to n-2 to avoid out-of-bounds access for s[i+1].
    if (s[i] == s[i + 1]) { // Checks if the current stone and the next one have the same color.
      c++; // If they are the same, increment the counter.
    }
  }

  cout << c << endl; // Prints the total count of stones that need to be removed.

  return 0; // Indicates successful program execution.
}
