/* Problem Statements
*  A. Petya and Strings
time limit per test2 seconds
memory limit per test256 megabytes
inputstandard input
outputstandard output

Little Petya loves presents.
His mum bought him two strings of the same size for his birthday.
The strings consist of uppercase and lowercase Latin letters.
Now Petya wants to compare those two strings lexicographically.
The letters' case does not matter, that is an uppercase letter is considered
equivalent to the corresponding lowercase letter. Help Petya perform the
comparison.

Input
Each of the first two lines contains a bought string.
The strings' lengths range from 1 to 100 inclusive.
It is guaranteed that the strings are of the same length and also consist of
uppercase and lowercase Latin letters.

Output
If the first string is less than the second one, print "-1".
If the second string is less than the first one, print "1".
If the strings are equal, print "0".
Note that the letters' case is not taken into consideration when the strings are
compared.

Examples
input
aaaa
aaaA
output
0
input
abs
Abz
output
-1
input
abcdefg
AbCdEfF
output
1
Note
If you want more formal information about the lexicographical order (also known
as the "dictionary order" or "alphabetical order"), you can visit the following
site:

http://en.wikipedia.org/wiki/Lexicographical_order
*/

/*
 *  Author   : Byungwook Lee
 *  Date     : 2016/10/25
 *  E-mail   : quddnr153@gmail.com
 *  Github   : github.com/quddnr153
 *  Filename : 112A_petyaandstrings.cpp
 */

#include <iostream>
#include <string.h>
#include <string>

using namespace std;

int main() {
  string first, second;
  cin >> first >> second;
  for (int i = 0; i < first.size(); i++) {
    if (first[i] - 'A' <= 26)
      first[i] += ('a' - 'A');
    if (second[i] - 'A' <= 26)
      second[i] += ('a' - 'A');
  }
  if (first.compare(second) == 0)
    cout << 0 << endl;
  else if (first > second)
    cout << 1 << endl;
  else
    cout << -1 << endl;
  return 0;
}
