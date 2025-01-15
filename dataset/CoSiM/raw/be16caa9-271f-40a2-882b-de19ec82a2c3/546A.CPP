#include "bits/stdc++.h"
using namespace std;

int main(int argc, char const *argv[]) {
  int k, n, w;
  cin >> k >> n >> w;

  int temp = k * (w * (w + 1) / 2);
  if (n >= temp)
    cout << '0';
  else
    cout << temp - n;
  return 0;
}
