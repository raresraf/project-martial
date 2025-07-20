#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main()

{
  int k;
  long long int n, w;
  cin >> k >> n >> w;
  int x = 0;

  for (int i = 1; i <= w; i++) {
    x = x + (k * i);
  }
  if (n >= x) {
    cout << "0";
  } else {
    cout << x - n;
  }

  return 0;
}
