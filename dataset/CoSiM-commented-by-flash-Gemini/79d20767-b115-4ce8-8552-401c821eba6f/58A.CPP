#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, m;
  // n red m blue
  cin >> n >> m;

  if (n == m) {
    cout << n << " " << 0 << endl;
  } else if (n > m && m == 0) {
    cout << 0 << " " << n / 2 << endl;
  } else if (m > n && n == 0) {
    cout << 0 << " " << m / 2 << endl;
  } else if (n > m && (n % 2 == 0) && (m % 2 == 0)) {
    cout << m << " " << (n - m) / 2 << endl;
  } else if (m > n && (n % 2 == 0) && (m % 2 == 0)) {
    cout << n << " " << (m - n) / 2 << endl;
  } else if (n > m && (n % 2 == 1) && (m % 2 == 1)) {
    cout << m << " " << (n - m) / 2 << endl;
  } else if (m > n && (n % 2 == 1) && (m % 2 == 1)) {
    cout << n << " " << (m - n) / 2 << endl;
  } else if (n > m && (n % 2 == 1) && (m % 2 == 0)) {
    int in = (n - m) / 2;
    cout << m << " " << in << endl;
  } else if (n > m && (n % 2 == 0) && (m % 2 == 1)) {
    int in = (n - m) / 2;
    cout << m << " " << in << endl;
  } else if (m > n && (n % 2 == 0) && (m % 2 == 1)) {
    int in = (m - n) / 2;
    cout << n << " " << in << endl;
  } else if (m > n && (n % 2 == 1) && (m % 2 == 0)) {
    int in = (m - n) / 2;
    cout << n << " " << in << endl;
  }
}
