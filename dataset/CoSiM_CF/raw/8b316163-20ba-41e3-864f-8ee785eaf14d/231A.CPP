#include <iostream>
using namespace std;
int main() {
  int i, n, a, b, c, result = 0;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> a >> b >> c;
    if ((a == 1 && b == 1) || (b == 1 && c == 1) || (c == 1) && (a == 1)) {
      result++;
    }
  }
  cout << result;

  return 0;
}
