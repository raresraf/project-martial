#include <cstring>
#include <iostream>

using namespace std;

int main() {
  int counter = 1, i = 1;

  string s;
  char a, b;

  cin >> s;

  a = s[0];

  for (int i = 0; i < s.length(); ++i) {
    b = s[i];

    if (b == a) {
      counter++;
    } else {
      counter = 1;
    }

    if (counter == 7) {
      cout << "YES" << endl;
      return 0;
    }
    a = b;
  }

  cout << "NO" << endl;

  return 0;
}