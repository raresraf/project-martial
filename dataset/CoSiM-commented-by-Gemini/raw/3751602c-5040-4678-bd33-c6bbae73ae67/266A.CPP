/**
 * @file 266A.CPP
 * @brief Solves the "Stones on the Table" problem.
 *
 * This program calculates the minimum number of stones to take from a row so
 * that no two adjacent stones have the same color. This is equivalent to
 * counting the number of adjacent pairs of identical characters in a string.
 *
 * Algorithm:
 * 1. Read the number of stones (n) and the string of colors (s).
 * 2. Initialize a counter for removals to 0.
 * 3. Iterate through the string, comparing each character with the one
 *    immediately following it.
 * 4. If a pair of adjacent characters are the same, increment the counter.
 * 5. Print the final count.
 *
 * Time Complexity: O(N), where N is the number of stones, as it requires a
 * single pass through the string.
 * Space Complexity: O(N) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

  /**
   * Functional Utility: This is a common preprocessor directive in competitive
   * programming to redirect standard input and output to files for easier
   * local testing. It has no effect when compiled with the ONLINE_JUDGE
   * flag, which is standard on most judging platforms.
   */
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  int n = 0; // Number of stones.
  int a = 0; // Counter for the number of stones to remove.
  cin >> n;
  string s;
  cin >> s;
  
  /**
   * Block Logic: Iterate through the string to find adjacent identical stones.
   * The loop runs up to the second-to-last character to allow for comparison
   * between s[i] and s[i+1].
   */
  for (int i = 0; i < n - 1; i++) {
    // Pre-condition: Check if the current stone and the next one have the same color.
    if (s[i] == s[i + 1]) {
      a += 1; // If they do, one must be removed.
    }
  }
  
  // Print the total number of removals required.
  cout << a << endl;

  return 0;
}
