#include <iostream>
using namespace std;
int x = 0;
int main() {
  string a;
  int n;
  cin >> n;
  for (int i = 1; i <= n; i++) {
    cin >> a;
    if (a == "++X" || a == "X++")
      x++;
    else if (a == "X--" || a == "--X")
      x--;
  }
  cout << x << endl;
  return 0;
}
