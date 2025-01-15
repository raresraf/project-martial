#include <iostream>
using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  int n;
  cin >> n;
  int a, b;
  cin >> a >> b;
  int x = 0;
  int y = -1;
  while (x * a <= n) {
    if ((n - x * a) % b == 0) {
      y = (n - x * a) / b;
      break;
    }
    ++x;
  }
  if (y == -1) {
    cout << "NO";
  } else {
    cout << "YES\n";
    cout << x << " " << y;
  }
  return 0;
}
