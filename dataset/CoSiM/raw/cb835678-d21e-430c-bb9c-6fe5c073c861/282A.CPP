// CodeForces
// Problem 282A [Bit++]

#include <iostream>
using namespace std;

int main() {
  int x = 0, t;
  char str[10];

  cin >> t;
  while (t--) {
    cin >> str;
    if (str[1] == '+') {
      x++;
    } else if (str[1] == '-') {
      x--;
    }
  }
  cout << x;
}