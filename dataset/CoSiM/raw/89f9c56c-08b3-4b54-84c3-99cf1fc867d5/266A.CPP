#include <iostream>
using namespace std;

int main() {
  int i, n, ct;
  char a, b;
  cin >> n;
  cin >> a;
  for (i = 2; i <= n; i++) {
    cin >> b;
    if (b == a) {
      ct++;
    }
    a = b;
  }
  cout << ct;
  return 0;
}
