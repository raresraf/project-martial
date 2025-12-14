#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  int x[n], y[n], z[n];
  for (int i = 0; i < n; i++) {
    cin >> x[i] >> y[i] >> z[i];
  }
  int a, b, c;
  a = 0;
  b = 0;
  c = 0;
  for (int i = 0; i < n; i++) {
    a = a + x[i];
    b = b + y[i];
    c = c + z[i];
  }
  if (a == 0) {
    if (b == 0 && c == 0) {
      cout << "YES" << endl;
    }
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
