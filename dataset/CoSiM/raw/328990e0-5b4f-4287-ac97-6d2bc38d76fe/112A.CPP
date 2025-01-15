#include <iostream>

using namespace std;

int main() {
  string a;
  string b;
  cin >> a;
  cin >> b;

  for (int i = 0; i < a.length(); i++) {
    if (a[i] > 64 && a[i] < 91) {
      a[i] += 32;
    }

    if (b[i] > 64 && b[i] < 91) {
      b[i] += 32;
    }
  }

  if (a < b) {
    cout << -1 << "\n";
  } else if (a > b) {
    cout << 1 << "\n";
  } else if (a == b) {
    cout << 0 << "\n";
  }
  return 0;
}