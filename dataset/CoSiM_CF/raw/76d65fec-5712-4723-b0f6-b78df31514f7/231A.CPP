#include <cctype>
#include <iostream>
#include <string>

using namespace std;

int main() {
  // freopen("input.txt", "r", stdin);
  int n;
  cin >> n;
  int cnt = 0;
  while (n--) {
    int x, y, z;
    cin >> x >> y >> z;
    if (x + y + z >= 2)
      ++cnt;
  }
  cout << cnt;
  return 0;
}
