/**
 * @file 58A.CPP
 * @brief This program determines if the word "hello" can be formed as a subsequence from a given input string.
 * A subsequence does not require consecutive characters, but they must appear in the correct order.
 * Algorithm: Brute-force approach using five nested loops to check all possible combinations of five characters
 * from the input string that could form the "hello" subsequence.
 * Time Complexity: O(N^5) where N is the length of the input string. This is highly inefficient but directly
 * reflects the implemented brute-force strategy.
 * Space Complexity: O(N) for storing the input string.
 */

#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience.

using namespace std;

int main() {
  string a; // Stores the input string.
  // Reads the input string from standard input.
  cin >> a;
  /**
   * Block Logic: Implements a brute-force search for the subsequence "hello".
   * Five nested loops iterate through all possible combinations of five indices (i, j, k, l, m)
   * such that i < j < k < l < m. These indices represent the positions of 'h', 'e', 'l', 'l', 'o'
   * in the input string, respectively.
   * If a matching subsequence is found, "YES" is printed, and the program terminates.
   * Invariant: The loops maintain the order of characters required for a subsequence.
   */
  for (int i = 0; i < a.length(); i++)
    for (int j = i + 1; j < a.length(); j++)
      for (int k = j + 1; k < a.length(); k++)
        for (int l = k + 1; l < a.length(); l++)
          for (int m = l + 1; m < a.length(); m++)
            // Checks if the characters at the selected indices match 'h', 'e', 'l', 'l', 'o' in sequence.
            if (a[i] == 'h' && a[j] == 'e' && a[k] == 'l' && a[l] == 'l' &&
                a[m] == 'o') {
              cout << "YES" << endl; // Prints "YES" if the subsequence is found.
              return 0;             // Terminates the program early upon finding the subsequence.
            }
  // If the loops complete without finding the "hello" subsequence, "NO" is printed.
  cout << "NO" << endl;
  // Standard program exit.
  return 0;
}
