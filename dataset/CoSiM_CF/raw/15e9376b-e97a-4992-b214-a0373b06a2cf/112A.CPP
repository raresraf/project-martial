#include <iostream>
#include <string>
using namespace std;

string lower_case(string word) {
  string new_word = "";
  for (short i = 0; i < word.size(); i++) {
    new_word += tolower(word[i]);
  }

  return new_word;
}

int main() {

  string a, b;
  cin >> a;
  cin >> b;

  a = lower_case(a);
  b = lower_case(b);

  if (a > b) {
    cout << 1;
  } else if (a < b) {
    cout << -1;
  } else {
    cout << 0;
  }
}