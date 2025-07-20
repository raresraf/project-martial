#include <bits/stdc++.h>

using namespace std;

int main() {
  int n = 0, num = 0;
  int x = 0, y = 0, z = 0;

  cin >> n;

  while (n--) {
    cin >> num;
    x += num;
    cin >> num;
    y += num;
    cin >> num;
    z += num;
  }

  if (x == 0 && y == 0 && z == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}