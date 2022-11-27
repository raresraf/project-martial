#include <bits/stdc++.h>
using namespace std;

int main() {

  int x;
  cin >> x;
  int y = x - 2;

  if (x % 2 == 0 && y != 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
