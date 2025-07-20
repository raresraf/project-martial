#include <bits/stdc++.h>

using namespace std;

int main() {
  string operation;
  int st = 0;
  int x = 0;

  cin >> st;

  while (st--) {
    cin >> operation;
    if ((operation[0] == '+') || (operation[2] == '+'))
      ++x;
    else if ((operation[0] == '-') || (operation[2] == '-'))
      --x;
  }

  cout << x;

  return 0;
}