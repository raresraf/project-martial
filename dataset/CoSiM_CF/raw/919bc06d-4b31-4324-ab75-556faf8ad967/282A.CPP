#include <iostream>
using namespace std;

int main() {
  int n, i, ct = 0;
  string a;
  cin >> n;
  for (i = 1; i <= n; i++) {
    cin >> a;
    if (a[1] == '+')
      ct++;
    else
      ct--;
  }
  cout << ct;
  return 0;
}
