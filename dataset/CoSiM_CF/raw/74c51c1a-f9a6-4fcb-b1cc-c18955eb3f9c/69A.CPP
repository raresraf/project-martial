#include <bits/stdc++.h>
#define LOG(x) cout << x << "\n"

using namespace std;

int main() {
  int n;
  cin >> n;
  int countX = 0;
  int countY = 0;
  int countZ = 0;
  while (n--) {
    int x, y, z;
    cin >> x >> y >> z;
    countX += x;
    countY += y;
    countZ += z;
  }
  if (countX == 0 && countY == 0 && countZ == 0) {
    LOG("YES");
  } else {
    LOG("NO");
  }
  return 0;
}