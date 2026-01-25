#include <iostream>
using namespace std;
int main() {
  int n, x = 0;
  cin >> n;
  string opr;
  for (int i = 0; i < n; i++) {
    cin >> opr;
    if (opr.at(1) == '+')
      x++;
    if (opr.at(1) == '-')
      x--;
  }
  cout << x << endl;

  return 0;
}
