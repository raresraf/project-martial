/* Petya started to attend programming lessons. On the first lesson his task was
to write a simple program. The program was supposed to do the following: in the
given string, consisting if uppercase and lowercase Latin letters, it:

1. deletes all the vowels,
2. inserts a character "." before each consonant,
3. replaces all uppercase consonants with corresponding lowercase ones.
4. Vowels are letters "A", "O", "Y", "E", "U", "I", and the rest are consonants.
The program's input is exactly one string, it should return the output as a
single string, resulting after the program's processing the initial string.

Help Petya cope with this easy task.

Input
The first line represents input string of Petya's program. This string only
consists of uppercase and lowercase Latin letters and its length is from 1 to
100, inclusive.

Output
Print the resulting string. It is guaranteed that this string is not empty. */

#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  cin >> s;
  transform(s.begin(), s.end(), s.begin(), ::tolower);
  string t = "";
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y')
      continue;
    else
      t = t + "." + s[i];
  }
  cout << t << endl;
  return 0;
}