#include <cctype>
#include <iostream>
#include <string>

using namespace std;

int main() {
  // freopen("input.txt", "r", stdin);
  int n;
  cin >> n;
  char x, y;
  int cnt = 0;
  cin >> x;
  while (cin >> y) {
    if (y != x)
      x = y;
    else
      cnt++;
  }
  cout << cnt;
  return 0;
}
