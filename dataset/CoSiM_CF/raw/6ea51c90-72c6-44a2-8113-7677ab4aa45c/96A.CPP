/* Petya loves football very much. One day, as he was watching a football match,
he was writing the players' current positions on a piece of paper. To simplify
the situation he depicted it as a string consisting of zeroes and ones. A zero
corresponds to players of one team; a one corresponds to players of another
team. If there are at least 7 players of some team standing one after another,
then the situation is considered dangerous. For example, the situation
00100110111111101 is dangerous and 11110111011101 is not. You are given the
current situation. Determine whether it is dangerous or not.

Input
The first input line contains a non-empty string consisting of characters "0"
and "1", which represents players. The length of the string does not exceed 100
characters. There's at least one player from each team present on the field.

Output
Print "YES" if the situation is dangerous. Otherwise, print "NO". */

#include <iostream>
#include <string>

using namespace std;

int main() {
  string pos;
  cin >> pos;
  if (pos.find("0000000") != string::npos ||
      pos.find("1111111") != string::npos)
    cout << "YES \n";
  else
    cout << "NO \n";
  return 0;
}