#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  int i = 0, n = 0, counter = 0;

  cin >> s;

  while (i < s.length()) {
    if (s[i] == 'h' && counter == 0) {
      counter = 1;
    } else if (counter == 1 && s[i] == 'e') {
      counter = 2;
    } else if (counter == 2 && s[i] == 'l') {
      counter = 3;
    } else if (counter == 3 && s[i] == 'l') {
      counter = 4;
    } else if (counter == 4 && s[i] == 'o') {
      counter = 5;
    }

    i++;
  }

  if (counter == 5) {
    cout << "YES" << endl;
    return 0;
  }

  cout << "NO" << endl;

  return 0;
}