#include <iostream>
using namespace std;

int main() {
  int r, c;
  for (int i = 0; i < 5; i++) {
    int x;
    for (int j = 0; j < 5; j++) {
      cin >> x;
      if (x == 1) {
        r = i;
        c = j;
      }
    }
  }
  cout << r << c;
  if (r <= 2)
    r = 2 - r;
  else
    r = r - 2;
  if (c <= 2)
    c = 2 - c;
  else
    c = c - 2;
  cout << c + r;
  return 0;
}