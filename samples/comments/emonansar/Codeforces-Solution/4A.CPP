#include <bits/stdc++.h>
using namespace std;
int main() {
  int w;
  cin >> w;
  if (w >= 1 && w <= 100) {
    if (w == 2) {
      cout << "NO" << endl;
    } else if (w % 2 == 0) {
      cout << "YES" << endl;
    } else {
      cout << "NO" << endl;
    }
  }
}
