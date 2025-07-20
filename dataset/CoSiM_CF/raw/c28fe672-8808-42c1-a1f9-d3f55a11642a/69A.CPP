//
// Created by Conor on 5/28/2020.
//

#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  int x = 0;
  int y = 0;
  int z = 0;

  int a, b, c;
  for (int i = 0; i < n; ++i) {
    cin >> a >> b >> c;
    x += a;
    y += b;
    z += c;
  }

  if (x == 0 && y == 0 && z == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }

  return 0;
}