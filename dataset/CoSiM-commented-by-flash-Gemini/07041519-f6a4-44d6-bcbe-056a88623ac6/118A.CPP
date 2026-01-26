// 118A - String Task

#include <iostream>

using namespace std;

int main() {
  string word;
  int c, i;

  cin >> word;
  c = word.size();

  for (i = 0; i < c; i++) {
    word[i] = char(tolower(word[i]));

    if (word[i] != 'a' && word[i] != 'e' && word[i] != 'i' && word[i] != 'o' &&
        word[i] != 'u' && word[i] != 'y')
      cout << '.' << word[i];
  }

  return 0;
}
