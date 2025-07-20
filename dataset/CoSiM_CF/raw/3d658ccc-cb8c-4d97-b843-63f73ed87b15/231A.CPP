

#include <ctype.h>
#include <iostream>
using namespace std;

int main() {
  int n;
  int a, b, c;
  cin >> n;
  int t = 0;
  for (int i = 0; i < n; i++) {
    cin >> a >> b >> c;
    if ((a + b + c) >= 2)
      t++;
  }
  cout << t;
}
