#include <iostream>

using namespace std;

int main() {
  int n, cnt = 0;
  cin >> n;
  while (n--) {
    int a, b, c;
    cin >> a >> b >> c;
    if ((a + b + c) >= 2)
      cnt++;
  }
  cout << cnt << endl;

  return 0;
}
