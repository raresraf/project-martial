// 28.01.19 @odnaks
#include <iostream>

using namespace std;

int main() {
  int k;
  int n;
  int w;
  int d;

  cin >> k >> n >> w;
  d = (k + w * k) * w / 2;
  if (d > n)
    d = d - n;
  else
    d = 0;
  cout << d;
  return (0);
}