/*
Link: https://codeforces.com/problemset/problem/71/A
@uthor: chavodon
*/

#include <iostream>
#include <sstream>

using namespace std;

int main() {
  string word, longWord, newWord;
  int i, n, count = 0, letterCount;
  char f, l;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> word;
    longWord += word + '!';
  }
  while (longWord[count] != '\0') {
    letterCount = 0;
    newWord = "";
    f = longWord[count];
    while (longWord[count + letterCount] != '!') {
      newWord += longWord[count + letterCount];
      letterCount++;
    }
    count += letterCount + 1;
    l = longWord[count - 2];
    if (letterCount > 10) {
      stringstream ss;
      ss << letterCount - 2;
      string s;
      ss >> s;
      newWord = f + s + l;
    }
    cout << newWord << endl;
  }
  return 0;
}
