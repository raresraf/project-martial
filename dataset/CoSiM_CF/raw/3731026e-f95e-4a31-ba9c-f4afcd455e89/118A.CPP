#include <algorithm>
#include <cstdio>
#include <iostream>
#include <set>
#include <string>
using namespace std;

int main(int argc, char *argv[]) {
  set<char> vowels;
  vowels.insert('a');
  vowels.insert('e');
  vowels.insert('i');
  vowels.insert('o');
  vowels.insert('u');
  vowels.insert('y');

  string s;
  cin >> s;
  transform(s.begin(), s.end(), s.begin(), ::tolower);
  for (int i = 0; i < s.length(); i++) {
    char c = s.at(i);
    // if character not exists in vowels
    if (vowels.find(c) == vowels.end())
      printf(".%c", c);
  }
  return 0;
}
