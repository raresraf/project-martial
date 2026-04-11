/**
 * @file 266A.CPP
 * @brief This program attempts to calculate a value based on a sequence of characters (stone colors).
 * It reads 'n' characters, then counts the number of adjacent pairs of characters that are *different*.
 * Finally, it outputs the total number of characters 'n' minus this count.
 * This does NOT solve the common problem of finding minimum removals for non-identical adjacent characters.
 *
 * Algorithm: Character sequence processing and counting of adjacent differing characters.
 * Time Complexity: O(N) where N is the number of characters.
 * Space Complexity: O(N) to store the character array.
 */

#include <iostream>
using namespace std;

int main() {

  // Functional Utility: 'n' stores the total number of characters (stones).
  int n;
  // Functional Utility: 'count' accumulates the number of adjacent pairs of characters that are different.
  int count = 0;

  // Block Logic: Reads the number of characters 'n' from standard input.
  cin >> n;

  // Functional Utility: 'cl' is a C-style character array (VLA - Variable Length Array) to store the input characters.
  //                     Note: VLAs are a C99 feature and a GCC extension in C++, not standard C++.
  char cl[n];

  /**
   * Block Logic: Reads 'n' individual characters into the character array 'cl'.
   * Invariant: After each iteration, `cl[i]` holds the i-th character.
   */
  for (int i = 0; i < n; i++) {
    cin >> cl[i];
  }

  /**
   * Block Logic: Iterates through the character array to count adjacent differing characters.
   * Invariant: 'j' represents the current index of the character being examined. The loop
   *            checks 'cl[j]' and 'cl[j+1]'. This loop may read beyond the bounds of `cl`
   *            if `j+1` goes up to `n`, accessing `cl[n]` which is out of bounds for an array of size `n`.
   *            This is a potential runtime error for C++ arrays.
   */
  for (int j = 0; j < n; j++) {
    /**
     * Block Logic: Checks if the current character and the next character are different.
     * Pre-condition: 'j+1' should be a valid index.
     * Invariant: If adjacent characters are different, 'count' is incremented.
     */
    if (cl[j] != cl[j + 1]) { // Potential out-of-bounds access if j == n-1
      count++;
    }
  }
  // Output the total number of characters 'n' minus the 'count' of adjacent differing pairs.
  // This value is not the standard solution for minimum stone removals.
  cout << n - count << endl;

  return 0;
}
