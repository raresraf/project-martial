#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

typedef long long ll;
using namespace std;

int c[100];
int main() {
  int a, b, d = 0;
  ll sum1 = 0, sum2 = 0;
  cin >> a;
  for (int i = 0; i < a; i++) {
    cin >> b;
    sum1 += b;
    c[i] = b;
  }
  sum1 /= 2;
  sort(c, c + a);
  for (int i = a - 1; i >= 0; i--) {
    sum2 += c[i];
    d++;
    if (sum1 < sum2)
      break;
  }
  cout << d;
}
