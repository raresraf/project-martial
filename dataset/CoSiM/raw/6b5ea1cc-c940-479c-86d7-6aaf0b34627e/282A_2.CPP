#include <iostream>
using namespace std;
int main() {
  int n, x = 0;
  string state;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> state;
    if (state.find("++") != string::npos)
      x++;
    else
      x--;
  }
  cout << x;
  return 0;
}
