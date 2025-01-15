// Word Capitalization
//
// https://codeforces.com/problemset/problem/281/A

// Capitalization is writing a word with its first letter as a capital letter.
// Your task is to capitalize the given word.
//
// Note, that during capitalization all the letters except the first one remains
// unchanged.
//
// Input
// A single line contains a non-empty word. This word consists of lowercase and
// uppercase English letters. The length of the word will not exceed 103.
//
// Output
// Output the given word after capitalization.

#include <iostream>
#include <string>
using namespace std;

int main() {
  string str;
  cin >> str;
  str[0] = toupper(str[0]);
  cout << str << endl;

  return 0;
}
