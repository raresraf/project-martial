/* Problem Statements
*  A. Football
time limit per test2 seconds
memory limit per test256 megabytes
inputstandard input
outputstandard output

Petya loves football very much.
One day, as he was watching a football match, he was writing the players'
current positions on a piece of paper. To simplify the situation he depicted it
as a string consisting of zeroes and ones. A zero corresponds to players of one
team; a one corresponds to players of another team. If there are at least 7
players of some team standing one after another, then the situation is
considered dangerous. For example, the situation 00100110111111101 is dangerous
and 11110111011101 is not. You are given the current situation. Determine
whether it is dangerous or not.

Input
The first input line contains a non-empty string consisting of characters "0"
and "1", which represents players. The length of the string does not exceed 100
characters. There's at least one player from each team present on the field.

Output
Print "YES" if the situation is dangerous. Otherwise, print "NO".

Examples
input
001001
output
NO
input
1000000001
output
YES
*/

/*
 *  Author   : Byungwook Lee
 *  Date     : 2016/10/24
 *  E-mail   : quddnr153@gmail.com
 *  Github   : github.com/quddnr153
 *  Filename : 96A_football.cpp
 */

#include <iostream>
#include <string.h>
#include <string>

using namespace std;

int main() {
  string input;
  cin >> input;
  string team1 = "1111111";
  string team2 = "0000000";
  bool chk = false;
  if (input.size() < 7)
    chk = false;
  else {
    for (int i = 0; i <= input.size() - 7; i++) {
      if (input.compare(i, team1.size(), team1) == 0 ||
          input.compare(i, team2.size(), team2) == 0) {
        chk = true;
        break;
      }
    }
  }
  if (chk)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
