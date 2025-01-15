#include <ctype.h>
#include <iostream>
using namespace std;

int main() {
  string word = "";
  string wordAfter = "";
  cin >> word;
  for (int i = 0; i < word.length(); i++) {
    word[i] = tolower(word[i]);
  }
  for (int i = 0; i < word.length(); i++) {
    if (word[i] != 'a' && word[i] != 'o' && word[i] != 'y' && word[i] != 'e' &&
        word[i] != 'u' && word[i] != 'i') {
      wordAfter += '.';
      wordAfter += word[i];
    }
  }

  cout << wordAfter;
  return 0;
}
