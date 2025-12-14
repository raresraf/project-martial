
#include <cctype>
#include <cstring>
#include <iostream>

using namespace std;

bool isVowel(char c) {
  return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' ||
          c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' ||
          c == 'y' || c == 'Y');
}

int main() {

  string word;
  cin >> word;

  for (char c : word) {

    if (isVowel(c)) {
      continue;
    }

    cout << '.';

    if (isupper(c)) {
      char x = c + 32;
      cout << x;
    } else
      cout << c;
  }

  cout << endl;

  return 0;
}