#include <bits/stdc++.h>
using namespace std;
int main() {
  char x;
  while (cin >> x) {
    if (x >= 'A' && x <= 'Z')
      x = x + 32;
    if (x == 'a' || x == 'e' || x == 'i' || x == 'o' || x == 'u' || x == 'y')
      ;
    else
      cout << '.' << x;
  }
}