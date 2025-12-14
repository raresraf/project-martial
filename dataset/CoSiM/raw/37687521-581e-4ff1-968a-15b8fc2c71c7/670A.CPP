#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  if (n % 7 == 0) {
    cout << (n / 7) * 2 << " " << (n / 7) * 2 << "\n";
  } else if ((n + 1) % 7 == 0) {
    cout << (int)(n / 7) * 2 + 1 << " " << (int)((n / 7) * 2 + 2) << "\n";
  } else if (n % 7 >= 2) {
    cout << (int)(n / 7) * 2 << " " << (int)((n / 7) * 2 + 2) << "\n";
  } else if (n % 7 == 1) {
    cout << (int)(n / 7) * 2 << " " << (int)((n / 7) * 2 + 1) << "\n";
  }
  return 0;
}
