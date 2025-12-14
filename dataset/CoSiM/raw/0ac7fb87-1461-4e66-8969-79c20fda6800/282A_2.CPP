#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, x = 0;
  string operation;
  cin >> n;
  while (n--) {
    cin >> operation;
    if (operation[0] == 'X') {
      if (operation[1] == '+') {
        x++;
      } else {
        x--;
      }
    } else if (operation[2] == 'X') {
      if (operation[0] == '+') {
        x++;
      } else {
        x--;
      }
    }
  }
  cout << x;

  return 0;
}
