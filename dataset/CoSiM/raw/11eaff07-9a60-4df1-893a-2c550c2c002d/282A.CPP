#include <cstring>
#include <iostream>
using namespace std;

int main() {
  int x = 0, s, i, len;
  string temp;

  cin >> s;

  while (s--) {
    cin >> temp;
    len = temp.length();

    for (i = 0; i < len; i++) {
      if (temp[i] == '+') {
        x++;
        break;
      } else if (temp[i] == '-') {
        x--;
        break;
      }
    }
  }

  cout << x;
}
