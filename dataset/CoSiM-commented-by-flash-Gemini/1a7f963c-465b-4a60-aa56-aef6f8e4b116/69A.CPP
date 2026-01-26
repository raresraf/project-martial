#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

int main() {
  int a = 0, b = 0, c = 0, d, e, f, n;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> d >> e >> f;
    a += d;
    b += e;
    c += f;
  }
  if (a == 0 && b == 0 && c == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
