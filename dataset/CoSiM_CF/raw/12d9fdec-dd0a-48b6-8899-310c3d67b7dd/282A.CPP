#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;

  int x = 0;

  for (int i = 0; i < n; ++i) {
    bool neg = false;
    char ch;
    for (int i = 0; i < 3; ++i) {
      cin >> ch;
      if (ch == '-') {
        neg = true;
      }
    }

    x += neg ? -1 : 1;
  }

  cout << x << endl;
}
