/**
 * @file 58A.CPP
 * @brief Solution for Codeforces Problem 58A - "Chat room".
 * @details The program checks if a given string contains the subsequence "hello".
 * A subsequence is a sequence of characters that appear in the same order in the string,
 * but not necessarily consecutively.
 *
 * @author (unknown)
 *
 * @algorithm
 * The program uses a brute-force approach with five nested loops to find the "hello" subsequence.
 * - The first loop searches for the character 'h'.
 * - The second loop searches for 'e' at an index after 'h'.
 * - The third loop searches for the first 'l' at an index after 'e'.
 * - The fourth loop searches for the second 'l' at an index after the first 'l'.
 * - The fifth loop searches for 'o' at an index after the second 'l'.
 *
 * If all five characters are found in the correct order, the program prints "YES" and terminates.
 * If the loops complete without finding the full subsequence, it prints "NO".
 *
 * @complexity
 * Time Complexity: O(N^5), where N is the length of the input string. This is highly inefficient
 * but may pass for small constraints on N.
 * Space Complexity: O(N) to store the input string.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  string a;
  cin >> a;
  /**
   * Block Logic: These five nested loops perform a brute-force search for the subsequence "hello".
   * Each level of the loop attempts to find the next character of "hello" at an index
   * greater than the index of the previously found character.
   */
  for (int i = 0; i < a.length(); i++)     // Search for 'h'
    for (int j = i + 1; j < a.length(); j++) // Search for 'e' after 'h'
      for (int k = j + 1; k < a.length(); k++) // Search for 'l' after 'e'
        for (int l = k + 1; l < a.length(); l++) // Search for the second 'l'
          for (int m = l + 1; m < a.length(); m++) // Search for 'o' after the second 'l'
            if (a[i] == 'h' && a[j] == 'e' && a[k] == 'l' && a[l] == 'l' &&
                a[m] == 'o') {
              // If the complete subsequence is found, print YES and exit.
              cout << "YES" << endl;
              return 0;
            }
  // If the loops complete without finding the subsequence, print NO.
  cout << "NO" << endl;
  return 0;
}
