#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  vector<int> v;
  int a = 0, b = 0, c = 0, d = 0;
  while (n--) {
    int i;
    cin >> i;
    if (i == 1)
      a++;
    else if (i == 2)
      b++;
    else if (i == 3)
      c++;
    else
      d++;
    if (b == 2) {
      d++;
      b = 0;
    }
  }
  if (a < c)
    a == 0;
  else
    a -= c;
  if (b > 0) {
    if (a > 2)
      a -= 2;
    else
      a = 0;
  }
  int temp = a % 4;
  a = a / 4;
  if (temp > 0)
    a++;
  cout << a + b + c + d << endl;

  return 0;
}
