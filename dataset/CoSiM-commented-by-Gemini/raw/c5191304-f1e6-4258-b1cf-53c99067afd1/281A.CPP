#include <bits/stdc++.h>
using namespace std;

int main() {
  char w[1000];
  cin >> w;
  int a = w[0];
  if (w[0] >= 'a') {
    w[0] = a - 32;
  }
  cout << w << endl;

  return 0;
}
