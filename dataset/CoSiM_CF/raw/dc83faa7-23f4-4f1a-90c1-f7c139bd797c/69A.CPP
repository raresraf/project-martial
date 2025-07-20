// Codeforces
// 69A - Young physicist

#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;

  int a, b, c, x = 0, y = 0, z = 0;
  for (int i = 0; i < n; i++) {
    scanf("%d %d %d", &a, &b, &c);
    x += a;
    y += b;
    z += c;
  }

  if (x == 0 && y == 0 && z == 0)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
