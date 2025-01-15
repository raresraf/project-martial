#include <iostream>
using namespace std;

int main() {
  int n, a = 0, b = 0, c = 0;
  cin >> n;
  int x[n], y[n], z[n];
  for (int i = 0; i < n; i++) {
    cin >> x[i] >> y[i] >> z[i];
  }
  for (int i = 0; i < n; i++) {
    a += x[i];
    b += y[i];
    c += z[i];
  }
  if (a + b + c == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
