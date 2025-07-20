#include <algorithm>
#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  int cs, in = 0, max = 0;
  while (cin >> cs) {
    int a[cs], b[cs];
    in = 0;
    max = 0;
    for (int i = 0; i < cs; i++) {
      cin >> a[i] >> b[i];
      in = in + b[i] - a[i];
      if (in > max)
        max = in;
    }
    cout << max << endl;
  }
}
