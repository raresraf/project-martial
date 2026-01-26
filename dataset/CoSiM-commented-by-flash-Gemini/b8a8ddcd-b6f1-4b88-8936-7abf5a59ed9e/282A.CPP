#include <iostream>
using namespace std;

int main() {
  int n, x = 0;
  char c[30];
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> c;
    if (c[1] == '+')
      ++x;
    else if (c[1] == '-')
      --x;
  }
  cout << x;
  return 0;
}
