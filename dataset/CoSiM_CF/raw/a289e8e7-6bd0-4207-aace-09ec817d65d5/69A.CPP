#include <iostream>

using namespace std;

int main() {
  int n = 0, x = 0, y = 0, z = 0;
  int tx = 0, ty = 0, tz = 0;
  string res = "";
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    tx += x;
    ty += y;
    tz += z;
  }

  if (tx == 0 && ty == 0 && tz == 0) {
    res = "YES";
  } else {
    res = "NO";
  }
  cout << res;
  return 0;
}
