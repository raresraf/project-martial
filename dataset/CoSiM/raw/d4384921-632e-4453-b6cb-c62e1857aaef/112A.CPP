#include <iostream>
using namespace std;
int main() {
  string a, b;
  cin >> a >> b;
  for (int i = 0; i < a.length(); i++) {
    if (a[i] < 92) {
      a[i] += 32;
    }
    if (b[i] < 92) {
      b[i] += 32;
    }
  }
  if (a < b) {
    cout << -1;
  } else if (a > b) {
    cout << 1;
  } else if (a == b) {
    cout << 0;
  }
}