#include <iostream>
using namespace std;

int main() {
  int n, x = 0, ans = 0;
  cin >> n;
  string o;
  for (int i = 0; i < n; i++) {
    cin >> o;
    if (o == "X++" || o == "++X")
      x++;
    else if (o == "X--" || o == "--X")
      x--;
  }
  cout << x;
}