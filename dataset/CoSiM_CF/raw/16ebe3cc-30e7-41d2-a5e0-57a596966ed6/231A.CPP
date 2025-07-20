#include <iostream>
using namespace std;
int main() {
  int a, b, c, n, sol = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> a >> b >> c;
    if (a > 0 && b > 0)
      sol++;
    else if (a > 0 && c > 0)
      sol++;
    else if (b > 0 && c > 0)
      sol++;
    else
      continue;
  }
  cout << sol;
}
