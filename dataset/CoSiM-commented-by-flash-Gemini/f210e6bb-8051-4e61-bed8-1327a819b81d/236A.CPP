/**
 * @file 236A.CPP
 * @brief This program determines the gender of a user based on their username.
 * If the number of distinct characters in the username is even, the user is considered
 * female ("CHAT WITH HER!"). If the number of distinct characters is odd, the user is
 * considered male ("IGNORE HIM!").
 *
 * Algorithm: Character frequency counting using a hash map (std::map) to find distinct characters.
 * Time Complexity: O(L) where L is the length of the username string.
 * Space Complexity: O(K) where K is the number of distinct characters in the username (at most 26 for English alphabet).
 */

#include <bits/stdc++.h> // Includes iostream, string, map, etc.
using namespace std;

int main() {
  // Functional Utility: 's' stores the input username string.
  string s;
  // Block Logic: Reads the username string from standard input.
  cin >> s;

  // Functional Utility: 'in' is a map that stores the frequency of each character in the username.
  //                     It effectively counts distinct characters.
  map<char, int> in;
  /**
   * Block Logic: Iterates through each character of the username to populate the frequency map.
   * Invariant: After each iteration, the frequency of 's[i]' is updated in the map.
   */
  for (int i = 0; i < s.length(); i++) {
    in[s[i]]++;
  }

  // Functional Utility: 'x' stores the number of distinct characters found in the username,
  //                     which is equivalent to the size of the map.
  int x = in.size();

  /**
   * Block Logic: Determines the gender based on the parity of the number of distinct characters.
   * Pre-condition: 'x' contains the count of distinct characters.
   * Invariant: If 'x' is even, prints "CHAT WITH HER!"; if odd, prints "IGNORE HIM!".
   */
  if (x % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
  return 0;
}
