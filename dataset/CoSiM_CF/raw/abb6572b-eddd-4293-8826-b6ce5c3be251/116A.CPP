#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  int n, a, b, p = 0, q = 0;
  cin >> n;
  while (n--) {
    cin >> a >> b;
    p -= a;
    p += b;
    if (p > q) {
      q = p;
    }
  }
  cout << q << endl;
}
