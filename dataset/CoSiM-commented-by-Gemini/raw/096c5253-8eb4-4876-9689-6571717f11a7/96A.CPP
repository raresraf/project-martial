// Football
//
//
// https://codeforces.com/problemset/problem/96/A
//
// Petya loves football very much. One day, as he was watching a football match,
// he was writing the players' current positions on a piece of paper. To simplify
// the situation he depicted it as a string consisting of zeroes and ones. A zero
// corresponds to players of one team; a one corresponds to players of another
// team. If there are at least 7 players of some team standing one after another,
// then the situation is considered dangerous. For example, the situation
// 00100110111111101 is dangerous and 11110111011101 is not. You are given the
// current situation. Determine whether it is dangerous or not.
//
// Input
// The first input line contains a non-empty string consisting of characters "0"
// and "1", which represents players. The length of the string does not exceed
// 100 characters. There's at least one player from each team present on the
// field.
//
// Output
// Print "YES" if the situation is dangerous. Otherwise, print "NO".

#include <iostream>

using namespace std;

/**
 * @brief This program determines if a given situation in a football match is "dangerous".
 *
 * A situation is considered dangerous if there are at least 7 consecutive players
 * from the same team. The team alignment is represented by a binary string, where
 * '0' and '1' represent players from the two different teams.
 *
 * The program reads a binary string from the standard input, iterates through it
 * to find the longest sequence of identical characters, and prints "YES" if this
 * sequence is 7 or longer, and "NO" otherwise.
 */
int main() {

  string str;
  cin >> str;

  int max = 0;
  int counter = 0;
  for (int i = 0; i < str.length() - 1; i++) {

    if (str[i] == str[i + 1]) {
      counter++;
      if (counter + 1 > max)
        max = counter + 1;

    } else {
      counter = 0;
    }
  }
  if (max > 6)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}