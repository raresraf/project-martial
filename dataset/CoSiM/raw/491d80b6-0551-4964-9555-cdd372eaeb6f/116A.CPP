#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  int n, enter, exit, result = 0, capasity = 0;

  cin >> n;

  while (n--) {
    cin >> exit >> enter;

    result = result + (enter - exit);

    if (result > capasity) {
      capasity = result;
    }
  }

  cout << capasity;

  return 0;
}
