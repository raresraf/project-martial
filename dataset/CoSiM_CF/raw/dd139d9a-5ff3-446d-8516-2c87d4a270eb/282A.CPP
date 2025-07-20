// Sagindyk Urazayev
// http://codeforces.com/problemset/problem/282/A
#include <iostream>
#include <string>

using namespace std;

int main() {
  string operation[150];
  int x, n, k;
  x = 0;
  cin >> n;
  for (k = 1; k < n + 1; k++)
    cin >> operation[k];

  for (n = 1; n < k + 1; n++) {
    if ((operation[n] == "X++") || (operation[n] == "++X"))
      x = x + 1;
    if ((operation[n] == "X--") || (operation[n] == "--X"))
      x = x - 1;
  }
  cout << x << endl;
}
