#include <bits/stdc++.h>

using namespace std;

int main() {

  int w;

  while (cin >> w && w > 0) {
    if (w % 2 == 0 && w > 2)
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  }

  return 0;
}