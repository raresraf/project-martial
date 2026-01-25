/**
 * @file 96A.CPP
 * @brief Solution to Codeforces problem 96A "Football".
 * 
 * @details
 * Problem Statement: A string represents a football match, with '0' indicating a player from
 * one team and '1' indicating a player from the other. A situation is considered "dangerous"
 * if there are at least 7 consecutive players of the same team. The program must determine
 * if the given situation is dangerous.
 *
 * Algorithm: The code iterates through the input string `s`. For each position `i`, it checks
 * if the substring of length 7 starting at `i` consists of identical characters. If such a
 * substring is found, it prints "YES" and terminates. If the entire string is scanned
 * without finding such a sequence, it prints "NO".
 *
 * Time Complexity: O(N), where N is the length of the string. The outer loop runs N-6 times
 * and the inner loop runs 7 times, resulting in a constant number of operations per character.
 * Space Complexity: O(N) to store the input string.
 */
// link : https://codeforces.com/contest/96/problem/A

#include <bits/stdc++.h>

using namespace std;

string s;

main() {

  cin >> s;

  // A dangerous situation is only possible if the string has at least 7 characters.
  if (s.size() >= 7) {
    /**
     * Block Logic: Iterate through all possible starting positions of a 7-character substring.
     */
    for (int i = 0; i < s.size() - 6; i++) {
      //	cout<<"1";
      int k = 0; // Counter for non-matching characters.
      char r = s[i]; // The character of the current potential sequence.

      /**
       * Block Logic: Check the 7 characters starting from position `i`.
       * Invariant: `k` counts how many characters in this 7-char window differ from the first one.
       */
      for (int j = i; j < i + 7; j++) {

        if (r != s[j]) {
          k++;
        }
      }

      // If k is 0, it means all 7 characters were identical.
      if (k == 0) {
        cout << "YES";
        return 0;
      }
    }
  }

  // If no sequence of 7 identical characters was found.
  cout << "NO";
  return 0;
}