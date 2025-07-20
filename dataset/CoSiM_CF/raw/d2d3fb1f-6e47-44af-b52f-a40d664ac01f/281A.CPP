#include <iostream>
#include <string>

using namespace std;

int Capitalize(char ch) {
  if (ch - 'a' < 0)
    return ch;
  return ch - 32;
}

int main() {
  string word;
  cin >> word;
  for (size_t i = 0; i < word.size(); ++i) {
    cout << (char)((i > 0) ? word[i] : Capitalize(word[i]));
  }
  return 0;
}
