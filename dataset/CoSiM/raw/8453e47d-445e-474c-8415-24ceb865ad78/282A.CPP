#include <iostream>

using namespace std;

int main() {
  int m, x = 0;
  cin >> m;
  string str;
  for (int i; i < m; i++) {
    cin >> str;
    if (str == "X++" || str == "++X")
      x++;
    if (str == "X--" || str == "--X")
      x--;
  }
  cout << x;
  return 0;
}
