#include <bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[]) {
  int n;
  cin >> n;

  int x = 0, y = 0, z = 0;
  for (int i = 0; i < n; ++i) {
    int a, b, c;
    cin >> a >> b >> c;
    x += a;
    y += b;
    z += c;
  }

  cout << (x == 0 && y == 0 && z == 0 ? "YES" : "NO") << endl;
}
