// by mo1ein
#include <iostream>
using namespace std;
int main() {
  int n = 0, X = 0;
  string x;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> x;
    if (x[0] == '-')
      --X;
    else if (x[0] == '+')
      ++X;
    else if (x[2] == '-')
      X--;
    else if (x[2] == '+')
      X++;
  }
  cout << X;
  return 0;
}
