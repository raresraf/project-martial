#include <iostream>
using namespace std;
int main() {
  int a, i, g = 0, h = 0, e = 0;
  cin >> a;
  int b[1000], c[1000], d[1000];
  for (i = 1; i <= a; i++) {
    cin >> b[i] >> c[i] >> d[i];
  }

  for (i = 1; i <= a; i++) {
    g += b[i];
  }

  for (i = 1; i <= a; i++) {
    h += c[i];
  }

  for (i = 1; i <= a; i++) {
    e += d[i];
  }
  if (g == 0 && h == 0 && e == 0) {
    cout << "YES";
  } else
    cout << "NO";
  return 0;
}
