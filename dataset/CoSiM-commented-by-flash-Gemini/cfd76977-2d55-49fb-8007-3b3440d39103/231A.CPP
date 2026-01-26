#include <bits/stdc++.h>
#define sc scanf
#define t1(t) scanf("%d", &t)
#define FOR(x) for (int i = 0; i < x; i++)
#define pf printf
#define wh(n) while (n--)
#define ll long long
using namespace std;

int main() {
  int t, count = 0;
  t1(t);
  wh(t) {
    int x, y, z;
    cin >> x >> y >> z;
    if (x && y || x && z || y && z)
      count++;
  }
  pf("%d\n", count);
}
