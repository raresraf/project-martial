/**
 * @file 58A.cpp
 * @brief Solution to a competitive programming problem: Chat room.
 * This program checks if the word "hello" can be spelled out by characters
 * in a given string, in the correct order (i.e., if "hello" is a subsequence
 * of the input string).
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Use a counter `cnt` as a state machine to track the letters of "hello".
 *    - `cnt` = 0: Looking for 'h'.
 *    - `cnt` = 1: Looking for 'e'.
 *    - `cnt` = 2: Looking for the first 'l'.
 *    - `cnt` = 3: Looking for the second 'l'.
 *    - `cnt` = 4: Looking for 'o'.
 * 3. Iterate through the input string. If the current character matches the
 *    expected character based on `cnt`, increment `cnt`.
 * 4. After iterating through the entire string, if `cnt` is 5, it means all
 *    letters of "hello" were found in order. Print "YES".
 * 5. Otherwise, print "NO".
 *
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(L) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string str;
  // cnt acts as a state machine for finding "hello".
  int cnt = 0, l;
  cin >> str;
  l = str.size();
  
  // Iterate through the input string to find the subsequence "hello".
  for (int i = 0; i < l; i++) {
    // State 0: Looking for 'h'.
    if (cnt == 0 && str[i] == 'h') {
      cnt++;
    } 
    // State 1: Looking for 'e'.
    else if (cnt == 1 && str[i] == 'e') {
      cnt++;
    } 
    // States 2 and 3: Looking for the two 'l's.
    else if (cnt >= 2 && cnt <= 3 && str[i] == 'l') {
      cnt++;
    } 
    // State 4: Looking for 'o'.
    else if (cnt == 4 && str[i] == 'o') {
      cnt++;
    }
  }
  
  // If cnt is 5, all characters of "hello" were found in order.
  if (cnt == 5) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
