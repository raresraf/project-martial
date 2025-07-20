#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
using namespace std;

int main() {
  long long n, a, b, s, r;
  freopen("B.txt", "r", stdin);
  while (cin >> n >> a >> b) {
    s = n * 6;
    r = a * b;
    while (s > r) {
      if (s % a == 0) {
        b = s / a;
      } else if (s % b == 0) {
        a = s / b;
      } else {
        if (a < b)
          b++;
        else
          a++;
      }
      r = a * b;
    }
    cout << a * b << "\n" << a << " " << b << endl;
  }
  return 0;
}
