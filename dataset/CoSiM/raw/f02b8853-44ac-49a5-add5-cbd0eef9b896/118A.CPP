/* Problem Statements
*  A. String Task
time limit per test2 seconds
memory limit per test256 megabytes
inputstandard input
outputstandard output

Petya started to attend programming lessons. On the first lesson his task was to
write a simple program. The program was supposed to do the following: in the
given string, consisting if uppercase and lowercase Latin letters, it:

deletes all the vowels,
inserts a character "." before each consonant,
replaces all uppercase consonants with corresponding lowercase ones.

Vowels are letters "A", "O", "Y", "E", "U", "I", and the rest are consonants.
The program's input is exactly one string, it should return the output as a
single string, resulting after the program's processing the initial string.

Help Petya cope with this easy task.

Input
The first line represents input string of Petya's program. This string only
consists of uppercase and lowercase Latin letters and its length is from 1 to
100, inclusive.

Output
Print the resulting string. It is guaranteed that this string is not empty.

Examples
input
tour
output
.t.r
input
Codeforces
output
.c.d.f.r.c.s
input
aBAcAba
output
.b.c.b
*/

/*
 *  Author   : Byungwook Lee
 *  Date     : 2016/10/24
 *  E-mail   : quddnr153@gmail.com
 *  Github   : github.com/quddnr153
 *  Filename : 118A_stringtask.cpp
 */

#include <iostream>
#include <string.h>
#include <string>

using namespace std;

int main() {
  string input;
  cin >> input;
  for (int i = 0; i < input.size(); i++) {
    if (input[i] == 'A' || input[i] == 'O' || input[i] == 'Y' ||
        input[i] == 'E' || input[i] == 'U' || input[i] == 'I')
      continue;
    if (input[i] == 'a' || input[i] == 'o' || input[i] == 'y' ||
        input[i] == 'e' || input[i] == 'u' || input[i] == 'i')
      continue;
    if (input[i] - 'A' >= 26)
      cout << '.' << input[i];
    else
      cout << '.' << (char)(input[i] + ('a' - 'A'));
  }
  cout << endl;
  return 0;
}