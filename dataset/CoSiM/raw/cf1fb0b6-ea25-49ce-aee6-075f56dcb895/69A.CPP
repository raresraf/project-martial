#include <bits/stdc++.h>

using namespace std;

#define lli long long int

lli mod = 1000000007;

void task() {
  int x = 0;
  int y = 0;
  int z = 0;

  int n;
  cin >> n;

  int fx, fy, fz;

  while (n--) {
    cin >> fx >> fy >> fz;

    x += fx;
    y += fy;
    z += fz;
  }

  if (x == 0 && y == 0 && z == 0) {
    cout << "YES\n";
  } else {
    cout << "NO\n";
  }
}

int main() { task(); }