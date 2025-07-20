

#include <iostream>

using namespace std;

int main() {
  int x = 0, n;
  cin >> n;
  string k;
  for (int i = 0; i < n; i++) {
    cin >> k;
    if ((k == "X++") || (k == "++X"))
      x++;
    if ((k == "X--") || (k == "--X"))
      x--;
  }
  cout << x;
  return 0;
}
