/**
 * @file 266A.CPP
 * @brief Solves the "Stones on the Table" problem from Codeforces.
 *
 * This program calculates the minimum number of stones to remove from a row
 * so that no two adjacent stones share the same color.
 *
 * @algorithm The code iterates through the string from the second character. It
 * maintains a state variable, `prev`, holding the character from the previous
 * position. If the current character is the same as `prev`, a counter is
 * incremented. If it's different, `prev` is updated to the current character.
 *
 * @time_complexity O(N), where N is the length of the string.
 * @space_complexity O(N) to store the input string.
 */
// 266A.cpp
// Stones on the Table
// Author: Ikki
// May 31, 2015

#include <iostream>
#include <string>
using namespace std;

int main() {
  int n;
  cin >> n;
  string str;
  cin >> str;
  int ans = 0;
  char prev = str[0];
  /**
   * @block_logic This loop iterates through the string, starting from the second
   * character, to count adjacent duplicates.
   */
  for (int i = 1; i < str.size(); i++) {
    // If the current character is the same as the one before it...
    if (str[i] == prev)
      ans++; // ...increment the count of stones to remove.
    else
      // Otherwise, update the previous character for the next comparison.
      prev = str[i];
  }
  cout << ans << endl;
  return 0;
}